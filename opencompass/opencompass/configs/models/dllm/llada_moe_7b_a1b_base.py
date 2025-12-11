from opencompass.models import LLaDAMoEModel

models = [
    dict(
        type=LLaDAMoEModel,
        abbr='llada-moe-7b-a1b-base',
        path='/TO/YOUR/PATH',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
