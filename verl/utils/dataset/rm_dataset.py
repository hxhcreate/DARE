# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

import verl.utils.torch_functional as verl_F
from verl.utils import hf_tokenizer
from omegaconf import ListConfig


def download_files_distributed(download_fn):
    import torch.distributed

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            # download files
            download_fn()

        torch.distributed.barrier()
    else:
        # download anyway
        download_fn()


class RMDataset(Dataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        max_length=1024,
        add_eos=True,
        cache_dir="~/.cache/verl/rm",
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self.add_eos = add_eos
        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        def _download_files():
            from verl.utils.fs import copy, is_non_local

            os.makedirs(self.cache_dir, exist_ok=True)
            assert os.path.exists(self.cache_dir)
            for i, parquet_file in enumerate(self.parquet_files):
                if is_non_local(parquet_file):
                    dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                    if not os.path.exists(dst):
                        copy(src=parquet_file, dst=dst)
                    self.parquet_files[i] = dst

        download_files_distributed(_download_files)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _pad_to_length(self, input_ids, attention_mask):
        curr_length = input_ids.shape[-1]

        if curr_length < self.max_length:
            input_ids = torch.cat((input_ids, torch.zeros(size=(self.max_length - curr_length,), dtype=input_ids.dtype)), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.zeros(size=(self.max_length - curr_length,), dtype=attention_mask.dtype)), dim=-1)
        elif curr_length > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return input_ids, attention_mask

    def __getitem__(self, item):
        prompt = self.prompts[item]
        chosen_response = self.chosen_responses[item]
        rejected_response = self.rejected_responses[item]
         
        prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        chosen_response = self.tokenizer.apply_chat_template(chosen_response, is_response=True, tokenize=False)
        rejected_response = self.tokenizer.apply_chat_template(rejected_response, is_response=True, tokenize=False)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        chosen_response_ids = self.tokenizer(chosen_response, return_tensors="pt")["input_ids"][0]
        rejected_response_ids = self.tokenizer(rejected_response, return_tensors="pt")["input_ids"][0]

        if self.add_eos:
            chosen_response_ids = torch.cat((chosen_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            rejected_response_ids = torch.cat((rejected_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)

        chosen_input_ids = torch.cat((prompt_ids, chosen_response_ids), dim=-1)
        chosen_attention_mask = torch.ones_like(chosen_input_ids)

        rejected_input_ids = torch.cat((prompt_ids, rejected_response_ids), dim=-1)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        chosen_input_ids, chosen_attention_mask = self._pad_to_length(chosen_input_ids, chosen_attention_mask)
        rejected_input_ids, rejected_attention_mask = self._pad_to_length(rejected_input_ids, rejected_attention_mask)

        input_ids = torch.stack((chosen_input_ids, rejected_input_ids), dim=0)
        attention_mask = torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class dLLMRMDataset(Dataset):
    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        max_prompt_length=512,
        max_response_length=512,
        add_eos=True,
        cache_dir="~/.cache/verl/rm",
        truncation="error",
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self.add_eos = add_eos
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        
        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        def _download_files():
            from verl.utils.fs import copy, is_non_local

            os.makedirs(self.cache_dir, exist_ok=True)
            assert os.path.exists(self.cache_dir)
            for i, parquet_file in enumerate(self.parquet_files):
                if is_non_local(parquet_file):
                    dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                    if not os.path.exists(dst):
                        copy(src=parquet_file, dst=dst)
                    self.parquet_files[i] = dst

        download_files_distributed(_download_files)

    # def _download(self, use_origin_parquet=False):
    #     from verl.utils.fs import copy_to_local

    #     data_files = self.data_files if not use_origin_parquet else self.original_data_files
    #     for i, parquet_file in enumerate(data_files):
    #         self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        chosen_response = self.chosen_responses[item]
        rejected_response = self.rejected_responses[item]
         
        prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        chosen_response = self.tokenizer.apply_chat_template(chosen_response, is_response=True, tokenize=False)
        rejected_response = self.tokenizer.apply_chat_template(rejected_response, is_response=True, tokenize=False)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        chosen_response_ids = self.tokenizer(chosen_response, return_tensors="pt")["input_ids"][0]
        rejected_response_ids = self.tokenizer(rejected_response, return_tensors="pt")["input_ids"][0]

        if self.add_eos:
            chosen_response_ids = torch.cat((chosen_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            rejected_response_ids = torch.cat((rejected_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)

        prompt_ids = prompt_ids.unsqueeze(0)
        chosen_response_ids = chosen_response_ids.unsqueeze(0)
        rejected_response_ids = rejected_response_ids.unsqueeze(0)
        
        prompt_input_ids, prompt_attention_mask = verl_F.postprocess_data(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        chosen_response_ids, chosen_response_attention_mask = verl_F.postprocess_data(
            input_ids=chosen_response_ids,
            attention_mask=torch.ones_like(chosen_response_ids),
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )
        
        rejected_response_ids, rejected_response_attention_mask = verl_F.postprocess_data(
            input_ids=rejected_response_ids,
            attention_mask=torch.ones_like(rejected_response_ids),
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )
        
        chosen_input_ids = torch.cat((prompt_input_ids[0], chosen_response_ids[0]), dim=-1)
        chosen_attention_mask = torch.cat((prompt_attention_mask[0], chosen_response_attention_mask[0]), dim=-1)
        
        rejected_input_ids = torch.cat((prompt_input_ids[0], rejected_response_ids[0]), dim=-1)
        rejected_attention_mask = torch.cat((prompt_attention_mask[0], rejected_response_attention_mask[0]), dim=-1)
        
        input_ids = torch.stack((chosen_input_ids, rejected_input_ids), dim=0)
        attention_mask = torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
