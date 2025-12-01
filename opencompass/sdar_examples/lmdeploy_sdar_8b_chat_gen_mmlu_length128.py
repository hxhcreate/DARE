from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets
    from ..opencompass.configs.models.dllm.lmdeploy_sdar_8b_chat import \
        models as lmdeploy_sdar_8b_chat
    from ..opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups
datasets = mmlu_datasets
models = lmdeploy_sdar_8b_chat
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

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
