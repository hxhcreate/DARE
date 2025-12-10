from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets
    from ..opencompass.configs.models.dllm.llada_moe_7b_a1b_instruct import \
        models as llada_moe_7b_a1b_instruct_models
    from ..opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups
datasets = mmlu_datasets
models = llada_moe_7b_a1b_instruct_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {
    'gen_blocksize': 3, 
    'gen_length': 3, 
    'gen_steps': 3, 
    'batch_size_': 1, 
    'batch_size': 1, 
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
        _scope_='opencompass',
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
