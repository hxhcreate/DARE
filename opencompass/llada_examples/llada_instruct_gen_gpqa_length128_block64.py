from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.gpqa.gpqa_fewshot_gen import \
        gpqa_datasets
    from ..opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
datasets = gpqa_datasets
models = llada_instruct_8b_models
eval_cfg = {
    'gen_blocksize': 64, 
    'gen_length': 128, 
    'gen_steps': 128, 
    'batch_size': 1, 
    'batch_size_': 1,
    'model_kwargs': {
        'attn_implementation': 'flash_attention_2',  #'sdpa'
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
    },
    'use_cache': True,
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
