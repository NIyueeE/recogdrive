set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

OUTPUT_DIR='qwenvl_chat/work_dirs/ReCogDrive_pretrain/qwen2.5_vl_7b_finetune_full_recogdrive_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=8 \
  --node_rank=$MLP_ROLE_INDEX \
  --master_addr=$MLP_WORKER_0_HOST \
  --nproc_per_node=${GPUS} \
  --master_port=$MLP_WORKER_0_PORT \
  qwenvl/train/qwenvl_chat_finetune.py \
  --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
  --conv_style "qwen" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --overwrite_output_dir True \
  --freeze_llm False \
  --freeze_vision False \
  --freeze_mlp False \
  --dataloader_num_workers 32 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 12288 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"