from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.hellaswag.hellaswag_gen_6faab5 import \
        hellaswag_datasets
    from ..opencompass.configs.models.dllm.lmdeploy_sdar_1dot7b_chat import \
        models as lmdeploy_sdar_1dot7b_chat
datasets = hellaswag_datasets
models = lmdeploy_sdar_1dot7b_chat

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
