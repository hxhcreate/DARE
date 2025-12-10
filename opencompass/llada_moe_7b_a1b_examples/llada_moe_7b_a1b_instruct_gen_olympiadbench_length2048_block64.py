from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_gen_be8b13 import \
        olympiadbench_datasets
    from ..opencompass.configs.models.dllm.llada_moe_7b_a1b_instruct import \
        models as llada_moe_7b_a1b_instruct_models
    from ..opencompass.configs.summarizers.OlympiadBench import summarizer

datasets = olympiadbench_datasets
models = llada_moe_7b_a1b_instruct_models

eval_cfg = {
    'gen_blocksize': 64, 
    'gen_length': 2048, 
    'gen_steps': 2048, 
    'batch_size': 1, 
    'batch_size_': 1,
    'model_kwargs': {
        'attn_implementation': 'flash_attention_2',  #'sdpa'
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
    },
    'use_cache': False,
}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    
        num_split=None,   
        min_task_size=16, 
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)

