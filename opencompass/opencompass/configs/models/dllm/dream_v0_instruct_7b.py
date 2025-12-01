from opencompass.models import DreamModel

models = [
    dict(
        type=DreamModel,
        abbr='dream-v0-instruct_7b',
        path='/TO/YOUR/PATH',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
