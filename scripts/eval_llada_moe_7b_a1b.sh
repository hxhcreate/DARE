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

model=${model:-LLaDA-MoE-7B-A1B-Instruct}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval.sh ${task}"
  echo "Optional task: mmlu, mmlupro, hellaswag, arcc, gsm8k math gpqa humaneval mbpp olympiadbench aime2024 aime2025"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=logs/EVAL/${exp_name}
mkdir -p ${log_dir}

# task Execution Map
case "${task}" in
  mmlu)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_mmlu_length256.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_mmlu_length256
    ;;
  mmlupro)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_mmlupro_length256.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_mmlupro_length256
    ;;
  hellaswag)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_hellaswag_length256.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_hellaswag_length256
    ;;
  gpqa)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_gpqa_length128.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_gpqa_length128
    ;;
  arcc)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_arcc_length512.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_arcc_length512
    ;;
  mbpp)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_mbpp_length512.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_mbpp_length512
    ;;
  humaneval)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_humaneval_length512.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_humaneval_length512
    ;;
  gsm8k)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_gsm8k_length256.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_gsm8k_length256
    ;;
  math)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_math_length512.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_math_length512
    ;;
  olympiadbench)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_olympiadbench_length2048.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_olympiadbench_length2048
    ;;
  aime2024)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_aime2024_length2048.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_aime2024_length2048
    ;;
  aime2025)
    py_script=llada_moe_7b_a1b_examples/llada_moe_7b_a1b_instruct_gen_aime2025_length2048.py
    work_dir=outputs/llada_moe_7b_a1b_instruct_gen_aime2025_length2048
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

python  run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err" 