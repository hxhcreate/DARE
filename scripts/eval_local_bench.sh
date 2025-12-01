#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# arguments parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      model="$2"
      shift; shift
      ;;
    --model_path)
      model_path="$2"
      shift; shift
      ;;
    --task)
      task="$2"
      shift; shift
      ;;
    --algorithm)
      algorithm="$2"
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

algorithm=${algorithm:-bgpo}
model=${model:-llada}
model_path=${model_path:-models/LLaDA-8B-Instruct}
engine=${engine:-hf}

if [ $task == "math" ]; then
    train_files="['data/preprocessed/train/math_1.parquet','data/preprocessed/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/test/math500_1.parquet','data/preprocessed/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=512
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
elif [ $task == "code" ]; then
    train_files="['data/preprocessed/train/lcbv5-K8_1.parquet','data/preprocessed/train/primeintellect-K8_1.parquet','data/preprocessed/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/test/mbpp_1.parquet','data/preprocessed/test/humaneval_1.parquet','data/preprocessed/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=512
    num_diffusion_steps=$max_response_length
    total_epoch=5
elif [ $task == "countdown" ]; then
    train_files="['data/preprocessed/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
elif [ $task == "sudoku" ]; then
    train_files="['data/preprocessed/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
fi

n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=16  # batch_size must be greater than the number of GPUs used
n_rollout=8
LR=5e-7
ppo_micro_batch_size_per_gpu=1  # gradient accumulation = batch_size / ppo_micro_batch_size_per_gpu
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
train_temperature=0.6
mask_token_id=126336
pad_token_id=126081

# diffusion related parameters
val_num_diffusion_steps=$max_response_length
block_length=32
mc_num=4
n_l=4
if [ $task == "code"]; then
    mc_num=2
    n_l=2
fi

project_name="EVAL_LOCAL_BENCH"
exp_name="${baseline}-btz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-step${num_diffusion_steps}-lr${LR}-temp${train_temperature}-n_l${n_l}-mc_num${mc_num}-gpu${n_gpus_per_node}"
model_path="./ckpts/LLaDA-8B-Instruct"  # NOTE: the path to the evaluted model

python3 -m src.dllm_main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=dllm \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    data.shuffle=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline=$baseline \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline=$baseline \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    +actor_rollout_ref.rollout.use_cache=True \
    +actor_rollout_ref.rollout.dual_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=$val_num_diffusion_steps \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.num_diffusion_steps=$num_diffusion_steps \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mc_num=$mc_num \
    +actor_rollout_ref.rollout.n_l=$n_l \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    +trainer.val_result_save_dir=$RESULT_SAVE_DIR \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$model_path \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm"
