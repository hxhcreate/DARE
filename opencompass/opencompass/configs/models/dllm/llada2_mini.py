from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada2-mini',
        path='/TO/YOUR/PATH',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
