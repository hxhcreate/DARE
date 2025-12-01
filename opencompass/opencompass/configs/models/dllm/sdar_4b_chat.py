from opencompass.models import SDARModel

models = [
    dict(
        type=SDARModel,
        abbr='sdar-4b-chat',
        path='/TO/YOUR/PATH',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
