#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

echo "[INFO] Cleaning up old Ray..."
ray stop --force || true
rm -rf /tmp/ray || true

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
    *)
      shift
      ;;
  esac
done

algorithm=${algorithm:-vrpo}
model=${model:-llada}
model_path=${model_path:-models/LLaDA-8B-Instruct}

# validate task
valid_tasks=("ultrafeedback")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

# validate model
valid_models=("llada" "dream" "sdar")
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '$model'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

# validate algorithm
valid_algorithms=("vrpo")
if [[ ! " ${valid_algorithms[@]} " =~ " ${algorithm} " ]]; then
    echo "Error: Invalid algorithm '$algorithm'"
    echo "Supported algorithms: ${valid_algorithms[*]}"
    exit 1
fi

# validate task
if [ $task == "ultrafeedback" ]; then
    train_files="['data/preprocessed/dpo/train/ultrafeedback.parquet']"
    val_files="['data/preprocessed/dpo/test/ultrafeedback.parquet']"
    max_prompt_length=512
    max_response_length=512
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=10
fi

# Set token IDs based on model
case $model in
    "llada")
        mask_token_id=126336
        pad_token_id=126081
        ;;
    "dream")
        mask_token_id=151666
        pad_token_id=151643
        ;;
    "sdar")
        mask_token_id=151669
        pad_token_id=151643
        ;;
    *)
        echo "Error: Unknown model '$model'"
        exit 1
        ;;
esac

# parameters setting
n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=64  # batch_size must be greater than the number of GPUs used
lr=5e-7
ppo_micro_batch_size_per_gpu=4  # must be even number to contain chosen and rejected samples
algorithm="vrpo"

ppo_mini_batch_size=$((n_gpus_per_node*ppo_micro_batch_size_per_gpu*2))

# diffusion related parameters
block_length=32
mc_num=4
n_l=4

beta=0.2

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
baseline="${model}-${algorithm}"
exp_name="${baseline}-bsz${batch_size}-prompt${max_prompt_length}-response${max_response_length}-lr${lr}-n_l${n_l}-mc_num${mc_num}-gpu${n_gpus_per_node}-${timestamp}"
ckpt_dir=./ckpts/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${ckpt_dir}
mkdir -p ${log_dir}

python3 -m verl.trainer.dllm_main_dpo \
    +algorithm.name=${algorithm} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=4 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.add_eos=false \
    data.truncation=left \
    +actor_rollout_ref.algorithm.name=${algorithm} \
    +actor_rollout_ref.model.name=$model \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.beta=$beta \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline=$baseline \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline=$baseline \
    +actor_rollout_ref.ref.mc_num=$mc_num \
    +actor_rollout_ref.ref.n_l=$n_l \
    +actor_rollout_ref.ref.cfg_scale=0.0 \
    +actor_rollout_ref.ref.baseline=$baseline \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=$total_epoch
    # >> ${log_dir}/${baseline}-${timestamp}.out \
    # 2>> ${log_dir}/${baseline}-${timestamp}.err &

# reward_model.reward_manager=dllm: used to select reward_manager in dllm_reward.load_reward_manager()
# llada does not support gradient_checkpointing
# custom_reward_function.name: stored as self.reward_fn, will be called using compute_reward() in ray_trainer

# Enable bfloat16
    # +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \

# Enable LoRA
    # actor_rollout_ref.model.lora_rank=1 \
    # actor_rollout_ref.model.lora_alpha=2 \
    # actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","ff_proj","up_proj","down_proj","gate_proj","ff_out"] \
    # +actor_rollout_ref.model.lora_dropout=0.05 \

# Disable dynamic batch size
    # actor_rollout_ref.actor.use_dynamic_bsz=False \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \

# Enable sequence parallelism, each GPU processes 1/4 of the sequence
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
