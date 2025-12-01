from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.aime2024.aime2024_gen_repeat8_6e39a4 import \
        aime2024_datasets
    from ..opencompass.configs.models.dllm.lmdeploy_sdar_4b_chat import \
        models as lmdeploy_sdar_4b_chat
    from ..opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchMath_summary_groups

datasets = aime2024_datasets
models = lmdeploy_sdar_4b_chat
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
summary_groups.extend([
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    }
])
summarizer = dict(
    dataset_abbrs=[
        # ['aime2024', 'accuracy'],
        ['AIME2024-Aveage8', 'naive_average'],
    ],
    summary_groups=summary_groups,
)

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

