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

import logging
import os
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from lmdeploy.utils import serialize_state_dict, FlattenedTensorBucket, FlattenedTensorMetadata
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from .base import BaseShardingManager
import requests
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPLMDeployShardingManager(BaseShardingManager):
    """FSDP <-> lmdeploy bridge."""

    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        base_url: str = "http://0.0.0.0:23333",
    ):
        self.module = module
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session = requests.Session()
        retries = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
        
        self.first_enter = True

    @GPUMemoryLogger(role="FSDPLMDeployShardingManager enter", logger=logger)
    def __enter__(self):  
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        device = torch.cuda.current_device() # used when fsdp2 set cpu_offload_policy
        params = {k: _preprocess_tensor_for_update_weights(v).to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        self._update_weights(params)
        log_gpu_memory_usage("After sync weights in sharding manager", logger=logger)
        del params

        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPLMDeployShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if dist.is_initialized():
            dist.barrier()

        self.module.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def _update_weights(self, state_dict: Dict[str, torch.Tensor]):
        is_rank_0 = True
        if dist.get_rank() != 0:
            is_rank_0 = False
        if is_rank_0:
            logger.info(f"Updating weights to LMDeploy API at {self.base_url}")

            # wake up weights, the server is ready for update weights
            # resp = requests.post(f"{self.base_url}/wakeup", headers=self.headers, json={"tags": ["weights", "kv_cache"]})
            # resp = self.session.post(f"{self.base_url}/wakeup", headers=self.headers, json={"tags": ["weights"]})
            # resp.raise_for_status()

            serialized_data = serialize_state_dict(state_dict)
            data = dict(serialized_named_tensors=serialized_data, finished=True)
            # resp = requests.post(f"{self.base_url}/update_weights", headers=self.headers, json=data)
            resp = self.session.post(f"{self.base_url}/update_weights", headers=self.headers, json=data, timeout=300)            
            resp.raise_for_status()

            # wake up kv cache (ready for inference)
            resp = self.session.post(f"{self.base_url}/wakeup", headers=self.headers, json={"tags": ["kv_cache"]})                        
            resp.raise_for_status()

        if dist.is_initialized():
            dist.barrier()

     
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()
        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]