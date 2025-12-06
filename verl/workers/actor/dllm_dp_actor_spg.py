# Copyright 2025 Shanghai AI Lab
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, kl_penalty, compute_policy_loss_SPG
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import DataParallelPPOActor
import torch.nn.functional as F

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)
        
        # diffusion related parameters
        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id
        self.mc_num = config["mc_num"]  # Number of Monte Carlo samples
        self.n_l = config["n_l"]  # Number of random masks
        self.cfg_scale = config["cfg_scale"]  # Whether to use CFG
        self.logp_estimation = config["logp_estimation"]
        
    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, logp_estimation="eubo", eubo_beta=1.0, mix_weight=0.5,  calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate log_probs and entropy for micro_batch
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            loss_per_sample: # (bs, mc_num)
        """
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seq_length - response_length
        device = micro_batch["input_ids"].device
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            # If there are multi-modal inputs, concatenate the content of each key
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        
        # Calculate log_probs
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]  # (bs, mc_num, seq_len)
            mask_indices = micro_batch["mask_indices"]  # (bs, mc_num, seq_len)
            p_mask = micro_batch["p_mask"]  # (bs, mc_num, seq_len)
            seq = micro_batch["input_ids"]  # (bs, seq_len)
            attention_mask = micro_batch["attention_mask"]  # (bs, mc_num, seq_len)

            loss_per_sample = torch.zeros((batch_size, mc_num), device=device)
            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]  # (batch_size, seq_len)
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]
                
                packed_perturbed_seq = []
                cu_seqlens = [0]
                max_seqlen = 0
                for b in range(batch_size):
                    valid_tokens = cur_perturbed_seq[b][attention_mask[b] == 1]
                    packed_perturbed_seq.append(valid_tokens)
                    cu_seqlens.append(cu_seqlens[-1] + len(valid_tokens))
                    max_seqlen = max(max_seqlen, len(valid_tokens))
                packed_perturbed_seq = torch.cat(packed_perturbed_seq, dim=0).unsqueeze(0)  # (1, total_seqlen)
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
                
                logits = self._get_logits(model=self.actor_module, packed_input=packed_perturbed_seq, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, prompt_len=attention_mask[:, :prompt_length].sum(dim=1), cfg_scale=0.0, MASK_TOKEN_ID=self.MASK_TOKEN_ID)
                
                # Restore logits for each sample
                loss = torch.zeros(batch_size, device=device)
                for b in range(batch_size):
                    start, end = cu_seqlens[b], cu_seqlens[b + 1]
                    
                    logits_b = torch.zeros(seq_length, logits.size(-1), device=device, dtype=logits.dtype)
                    logits_b[attention_mask[b] == 1] = logits[0, start:end]
                    
                    mask = cur_mask_indices[b]  # (seq_len,)
                    loss[b] = (F.cross_entropy(logits_b[mask], seq[b][mask], reduction="none") / cur_p_mask[b][mask]).sum()  # cross_entropy returns negative log likelihood
                loss_per_sample[:, i] = -loss  # convert to log likelihood

            log_likelihood = loss_per_sample.sum(dim=1) / mc_num  # (batch_size,)
            log_probs = (log_likelihood / response_length).view(-1, 1).repeat(1, response_length)  # (batch_size, response_length)
            loss_per_sample_expanded = (loss_per_sample / response_length).unsqueeze(-1).expand(-1, -1, response_length).contiguous()
            

        # Compute EUBO if needed
        if logp_estimation == 'eubo' or logp_estimation == 'mix':
            completion_mask_expanded = attention_mask.unsqueeze(1).expand(-1, mc_num, -1)[:, :, -response_length:]
            perturb_mask_expanded = (perturbed_seq == self.MASK_TOKEN_ID)[:, :, -response_length:]
            empirical_t = (completion_mask_expanded * perturb_mask_expanded).sum(dim=2) / completion_mask_expanded.sum(dim=2).clamp(min=1e-8)
            empirical_t_expanded = empirical_t.unsqueeze(2).expand(-1, -1, completion_mask_expanded.size(-1))
            # Compute per_token_avg_ps
            per_token_avg_ps = log_probs.pow(eubo_beta) * perturb_mask_expanded * completion_mask_expanded / empirical_t_expanded.clamp(min=1e-8)
            per_token_avg_ps = per_token_avg_ps.mean(dim=1)  # [batch_size, response_length]

            loss_mask = (per_token_avg_ps > 0).bool()
            
            # Handle zero values in per_token_avg_ps
            per_token_avg_ps_dezero = per_token_avg_ps.clone()
            per_token_avg_ps_dezero[per_token_avg_ps == 0] = 1
            del per_token_avg_ps
            
        
        # Calculate logp_positive and logp_negative
        logp_positive = log_probs
        if logp_estimation == "elbo":
            logp_negative = log_probs
        elif logp_estimation == "eubo":
            logp_negative = (
        (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=1, keepdim=True) / loss_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    ).repeat(1, response_length)  # [batch_size, response_length]
        elif logp_estimation == "mix":
            logp_negative = (
        mix_weight * (per_token_avg_ps_dezero.log() * loss_mask).sum(dim=1, keepdim=True) / loss_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        + (1 - mix_weight) * log_probs.mean(dim=1, keepdim=True)
    ).repeat(1, response_length)  # [batch_size, response_length]

        else:
            raise ValueError(f"Unsupported logp_estimation: {logp_estimation}")

        entropy = None
        if calculate_entropy:
            probs = log_probs.exp()
            entropy = -probs * log_probs  # (bs, response_length) entropy of each token
            
        return entropy, logp_positive, logp_negative, loss_per_sample_expanded


        
        entropy = None
        if calculate_entropy:
            probs = log_probs.exp()
            entropy = -probs * log_probs  # (bs, response_length) entropy of each token
            
        return entropy, log_probs, loss_per_sample            

    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size+1,)
        max_seqlen: int
        prompt_len: (batch_size,) True prompt length of each sample
        """
        # If CFG is used, fuse conditional and unconditional logits
        if cfg_scale > 0.:
            un_packed_input = packed_input.clone()
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                un_packed_input[0, start:start + prompt_len[i].item()] = MASK_TOKEN_ID  # mask prompt part and get unconditional input
            packed_input_cat = torch.cat([packed_input, un_packed_input], dim=0)  # concatenate conditional and unconditional input
            cu_seqlens_cat = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
            logits = model(packed_input_cat, cu_seqlens=cu_seqlens_cat, max_seqlen=max_seqlen).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)  # split by batch
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)  # fusion formula
        else:
            logits = model(packed_input, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
        return logits[:, :packed_input.shape[1]]
    

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "perturbed_seq", "mask_indices", "p_mask"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_positive_lst = []
        log_probs_negative_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, logp_positive, logp_negative, _ = self._forward_micro_batch(micro_batch, temperature=temperature, n_l=self.n_l, 
                                                                                     mc_num=self.mc_num, logp_estimation=self.logp_estimation,
                                                                                     calculate_entropy=calculate_entropy, call_fn_name="compute_log_prob")
            log_probs_positive_lst.append(logp_positive)
            log_probs_negative_lst.append(logp_negative)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs_positive = torch.concat(log_probs_positive_lst, dim=0)
        log_probs_negative = torch.concat(log_probs_negative_lst, dim=0)
        
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs_positive.size(0), f"{len(indices)} vs. {log_probs_positive.size()}"
            assert len(indices) == log_probs_negative.size(0), f"{len(indices)} vs. {log_probs_negative.size()}"
            
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs_positive = log_probs_positive[revert_indices]
            log_probs_negative = log_probs_negative[revert_indices]
            

        return log_probs_positive, log_probs_negative, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages", "perturbed_seq", "mask_indices", "p_mask"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()  # Clear gradient accumulation at the beginning of each micro-batch

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    entropy, log_prob, log_prob_negative, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature, n_l=self.n_l, 
                                                                                     mc_num=self.mc_num, logp_estimation=self.logp_estimation,
                                                                                     calculate_entropy=calculate_entropy, call_fn_name="update_policy")
                    
                    # entropy, log_prob, log_prob_negative, loss_per_sample = self._forward_micro_batch(micro_batch=data, temperature=temperature, n_l=self.n_l, mc_num=self.mc_num, calculate_entropy=calculate_entropy, call_fn_name="update_policy")              
                    
                    # use_SPG use RL loss
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_SPG(
                        log_prob_positive=log_prob,
                        log_prob_negative=log_prob_negative,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                    )
                    
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:  # NOTE: Currently not considering KL
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    print(f"loss: {loss}")
                    loss.backward()

                    data = {
                    "actor/pg_loss": pg_loss.detach().item(),
                    }
                    # data = {
                    #     "actor/pg_loss": pg_loss.detach().item(),
                    #     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    #     "actor/ppo_kl": ppo_kl.detach().item(),
                    #     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    # }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()  # Update gradients after each mini-batch
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()  # Clear gradient accumulation
        return metrics