#!/bin/bash
set -e

export TORCHDYNAMO_DISABLE=1
export HF_HOME=
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1
export COMPASS_DATA_CACHE=opencompass
cd opencompass



# parameter parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --task)
      task="$2"
      shift; shift
      ;;
    --model)
      model="$2"
      shift; shift
      ;;
    --engine)
      engine="$2"
      shift; shift
      ;;
    *)
      shift
      ;;
  esac
done

model=${model:-LLaDA-8B-Instruct}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval.sh ${task}"
  echo "Optional task: mmlu, mmlupro, hellaswag, arcc, gsm8k_confidence math_confidence gpqa_confidence humaneval_logits mbpp_confidence gsm8k_short math_short"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=logs/EVAL/${exp_name}
mkdir -p ${log_dir}

# task Execution Map
case "${task}" in
  mmlu)
    py_script=llada_examples/llada_instruct_gen_mmlu_length3_block3.py
    work_dir=outputs/llada_instruct_mmlu_length3_block3
    ;;
  mmlupro)
    py_script=llada_examples/llada_instruct_gen_mmlupro_length256_block256.py
    work_dir=outputs/llada_instruct_mmlupro_length256_block256
    ;;
  hellaswag)
    py_script=llada_examples/llada_instruct_gen_hellaswag_length3_block3.py
    work_dir=outputs/llada_instruct_hellaswag_length3_block3
    ;;
  arcc)
    py_script=llada_examples/llada_instruct_gen_arcc_length512_block512.py
    work_dir=outputs/llada_instruct_arcc_length512_block512
    ;;
  gsm8k_confidence)
    py_script=llada_examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py
    work_dir=outputs/llada_instruct_gsm8k_length512_block512_confidence
    ;;
  math_confidence)
    py_script=llada_examples/llada_instruct_gen_math_length512_block512_confidence.py
    work_dir=outputs/llada_instruct_math_length512_block512_confidence
    ;;
  gpqa_confidence)
    py_script=llada_examples/llada_instruct_gen_gpqa_length64_block64_confidence.py
    work_dir=outputs/llada_instruct_gen_gpqa_length64_block64_confidence
    ;;
  humaneval_logits)
    py_script=llada_examples/llada_instruct_gen_humaneval_length512_block512_logits.py
    work_dir=outputs/llada_instruct_gen_humaneval_length512_block512_logits
    ;;
  mbpp_confidence)
    py_script=llada_examples/llada_instruct_gen_mbpp_length256_block256_confidence.py
    work_dir=outputs/llada_instruct_gen_mbpp_length256_block256_confidence
    ;;
  gsm8k)
    py_script=llada_examples/llada_instruct_gen_gsm8k_length256_block8.py
    work_dir=outputs/llada_instruct_gen_gsm8k_length256_block8
    ;;
  math)
    py_script=llada_examples/llada_instruct_gen_math_length512_block64.py
    work_dir=outputs/llada_instruct_gen_math_length512_block64
    ;;
  olympiad)
    py_script=llada_examples/llada_instruct_gen_olympiadbench_length2048_block64.py
    work_dir=outputs/llada_instruct_gen_olympiadbench_length2048_block64
    ;;
  aime2024)
    py_script=llada_examples/llada_instruct_gen_aime2024_length2048_block64.py
    work_dir=outputs/llada_instruct_gen_aime2024_length2048_block64
    ;;
  aime2025)
    py_script=llada_examples/llada_instruct_gen_aime2025_length2048_block64.py
    work_dir=outputs/llada_instruct_gen_aime2025_length2048_block64
    ;;
  *)
    echo "Unknown task: ${task}"
    exit 1
    ;;
esac

echo "task: ${task}"
echo "model: ${model}"
echo "Script: ${py_script}"
echo "Work Dir: ${work_dir}"
echo "Log Dir: ${log_dir}"

python run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err" &
