from opencompass.models import LLaDA2Model

models = [
    dict(
        type=LLaDA2Model,
        abbr='llada2-mini',
        path='/TO/YOUR/PATH',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
