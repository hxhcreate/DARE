# Copyright 2025 Shanghai AI Lab Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A simple lmdeploy rollout implementation.
It uses lmdeploy.pipeline as the inference engine.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from ..rollout_utils import process_sdar_generation_outputs
from openai import OpenAI, APIConnectionError, APIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    # lmdeploy>=0.5 style
    from lmdeploy import PytorchEngineConfig, GenerationConfig
    from lmdeploy.pytorch.engine import Engine, EngineInstance
except ImportError:
    Engine = None
    EngineInstance = None
    GenerationConfig = None
    logger.warning("lmdeploy is not installed, LMDeployRollout will not work.")


def align_sequence(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    prompt_length: int,
    response_length: int, 
    max_new_tokens: int,
    attention_mask: torch.Tensor,
    position_ids,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
):
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        prompt: Input token indices [seq_len]
        response: Output token indices [seq_len]
        prompt_length: Input token indices length
        response_length: Output token indices length
        attention_mask: Mask [seq_len]
        max_length: Max sequence length
        pad_token_id: Padding token ID
        eos_token_id: EOS token ID
        truncation: "left", "right" or "middle"

    Returns:
        (sequence_ids, response_ids, attention_mask) padded/truncated to max_length
    """
    assert prompt_ids.ndim == 2

    sequence_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    # Update position_ids / attention_mask
    response_attention_mask = get_response_mask(
        response_id=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
    ) 
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    if prompt_length + response_length < max_length:
        sequence_ids = pad_sequence_to_length(sequence_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=False)
        response_ids = pad_sequence_to_length(response_ids, max_seq_len=max_new_tokens, pad_token_id=pad_token_id, left_pad=False)
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=False)
    elif prompt_length + response_length > max_length:
        sequence_ids = sequence_ids[:, -max_length:]
        response_ids = response_ids[:, -max_new_tokens:]
        attention_mask = attention_mask[:, -max_length:]

    # long_attention_mask = attention_mask.long()
    # position_ids = (torch.cumsum(long_attention_mask, dim=1) - 1) * long_attention_mask
    delta_position_id = torch.arange(1, max_new_tokens + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0)
    response_position_ids = position_ids[:, -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

    # Ensure position_ids is on the correct device
    if position_ids.device != sequence_ids.device:
        position_ids = position_ids.to(sequence_ids.device)
        
    return sequence_ids, response_ids, attention_mask, position_ids


class LMDeployRollout(BaseRollout):
    """A rollout engine based on lmdeploy.pytorch.engine Engine.

    - Receive verl.DataProto (input_ids, attention_mask, position_ids...)
    - Turn to lmdeploy format (list of token ids)
    - Turn to DataProto
    Returns:
        Don't support multi-turn / tool calls, only single-turn generation
        Don't support token-level logprob, only return responses
    """

    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        device_mesh,
        **kwargs,
    ):
        """
        Args:
            model_path: Huggingface model path to the model. The model should be supported by lmdeploy.
            config: A DictConfig object containing lmdeploy-specific operational
                parameters and rollout settings.
            tokenizer: The tokenizer instance compatible with the actor_module.
            model_hf_config: The huggingface config to initiallize the generating model in lmdeploy
            **kwargs: Reserved
        """
        super().__init__()
        
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # is_rank_0 = True
        # if dist.get_rank() != 0:
        #     is_rank_0 = False
        # if is_rank_0:
        base_url = os.getenv("LMDEPLOY_API_URL", "http://0.0.0.0:23333/v1")
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = self.client.models.list().data[0].id
        logger.info(f"LMDeployRollout initialized with API URL: {base_url}")

    @GPUMemoryLogger(role="lmdeploy rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """DataProto -> lmdeploy -> DataProto"""
        # Start timer
        start_time = time.time()
        # input_ids: (bs, prompt_length), left padding
        raw_prompt = prompts.non_tensor_batch["raw_prompt"] # np.array(list[dict[str]], dtype=object)
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)

        # used to construct attention_mask
        eos_token_id = prompts.meta_info.get("eos_token_id", self.eos_token_id)

        # parese idx from torch.Tensor to list[list[int]]
        raw_prompt_list = []
        for i in range(batch_size):
            raw_prompt_list.append(raw_prompt[i])

        is_validate = prompts.meta_info.get("validate", False)
        if is_validate:
            gen_kwargs = {
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "max_tokens": self.config.get("max_new_tokens", 4096),
            }
            n_rollout = self.config.val_kwargs.n  # if validate, already repeat in ray_trainer
        else:
            gen_kwargs = {
                "top_p": self.config.top_p,
                "temperature": self.config.temperature,
                "max_tokens": self.config.get("max_new_tokens", 4096),
            }
            n_rollout = self.config.n  # if validate, already repeat in ray_trainer
        print(f"gen_kwargs: {gen_kwargs}")

        messages = [prompt for prompt in raw_prompt_list for _ in range(n_rollout)]
        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)
        position_ids_repeat = position_ids.repeat_interleave(n_rollout, dim=0)
        total_batch_size = batch_size * n_rollout

        max_new_tokens = self.config.get("max_new_tokens", 4096)
        prompt_length = idx.size(1)
        max_length = prompt_length + max_new_tokens

        # parse output
        batch_sequence: List[torch.Tensor] = []
        batch_response: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_position_ids: List[torch.Tensor] = []

        for i in range(total_batch_size):
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages[i],
                        **gen_kwargs
                    )
                    break
                except (APIConnectionError, APIError, RateLimitError, APITimeoutError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"[Rank {dist.get_rank()}] Failed to get response for index {i} after {max_retries} attempts: {e}")
                        raise e
                    logger.warning(f"[Rank {dist.get_rank()}] Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in 2s...")
                    time.sleep(2)
            text= response.choices[0].message.content
            response_ids = self.tokenizer.encode(text, add_special_tokens=False)
            response_length = len(response_ids)
            response_ids = torch.tensor(response_ids, dtype=torch.long, device=idx.device)
            sequence_ids, response_ids, attention_mask_i, position_ids_i = align_sequence(
                prompt_ids=idx_repeat[i].unsqueeze(0), 
                response_ids=response_ids.unsqueeze(0), 
                prompt_length=prompt_length,
                response_length=response_length, 
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask_repeat[i].unsqueeze(0),
                position_ids=position_ids_repeat[i].unsqueeze(0),
                max_length=max_length, 
                pad_token_id=self.pad_token_id,
                eos_token_id=eos_token_id
            )
            batch_sequence.append(sequence_ids)
            batch_response.append(response_ids)
            batch_attention_mask.append(attention_mask_i)
            batch_position_ids.append(position_ids_i)
        sequences = torch.cat(batch_sequence, dim=0)
        responses = torch.cat(batch_response, dim=0)
        attention_mask = torch.cat(batch_attention_mask, dim=0)
        position_ids = torch.cat(batch_position_ids, dim=0) 

        answers = process_sdar_generation_outputs(
            outputs=sequences,
            idx_repeat=idx_repeat,
            tokenizer=self.tokenizer,
        )

        if not is_validate:
            for j in range(total_batch_size):
                print(f"==================[RANK{dist.get_rank()}] rollout question ID: {j}=================\nGenerated answer: {answers[j]}\n==========================================")
        else:
            for j in range(total_batch_size):
                print(f"==================[RANK{dist.get_rank()}] validation question ID: {j}=================\nGenerated answer: {answers[j]}\n==========================================")

        batch = TensorDict(
            {
                "prompts": idx_repeat,
                "responses": responses,
                "input_ids": sequences,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=total_batch_size,
        )
        total_time = time.time() - start_time
        print(f"[RANK{dist.get_rank()}] generate_sequences total time: {total_time:.2f}s")

        return DataProto(batch=batch)