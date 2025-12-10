# from opencompass.models import LLaDAModel
from opencompass.models import LLaDAMoeModel

models = [
    dict(
        type=LLaDAMoeModel,
        abbr='llada-moe-7b-a1b-instruct',
        path='/TO/YOUR/PATH',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
