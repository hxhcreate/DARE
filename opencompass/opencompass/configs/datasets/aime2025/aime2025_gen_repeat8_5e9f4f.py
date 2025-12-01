"""
Summary: A config for AIME-2025 Evaluation.
Setting:
    Shot: 0-shot
    Evaluator:
        - CascadeEvaluator
            - MATHVerifyEvaluator
            - GenericLLMEvaluator
    Repeat: 1
Avaliable Models:
    - Instruct/Chat Models
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset, MATHEvaluator, math_postprocess_v2
from opencompass.evaluator import MATHVerifyEvaluator

aime2025_reader_cfg = dict(input_columns=['question'], output_column='answer')


aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{question}\nRemember to put your final answer within \\boxed{}.',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2025_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2025_datasets = [
    dict(
        type=CustomDataset,
        abbr=f'aime2025-run{idx}',
        path='opencompass/aime2025',
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=aime2025_eval_cfg,
        n=1,
    )
    for idx in range(8)
]