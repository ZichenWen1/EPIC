#!/bin/bash


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' 

cd ~/EPIC

# Check for arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <method> [checkpoint_name]"
    echo "Methods: dart, fastv, random"
    echo "Example: $0 dart my_checkpoint"
    exit 1
fi

METHOD=$1
CKPT=${2:-"ckpt_${METHOD}_$(date +%Y%m%d_%H%M%S)"}

# Validate method argument
case $METHOD in
    dart|fastv|random)
        echo "Training with method: $METHOD"
        ;;
    *)
        echo "Error: Unsupported method '$METHOD'"
        echo "Supported methods: dart, fastv, random"
        exit 1
        ;;
esac


# Set different parameters based on the method
case $METHOD in
    dart)
        ALPHA=0.5
        ATTENTION="flash_attention_2"
        ;;
    fastv)
        ALPHA=0.5
        ATTENTION="sdpa"
        ;;
    random)
        ALPHA=0.4
        ATTENTION="flash_attention_2"
        ;;
esac

per_device_train_batch_size=16
per_device_eval_batch_size=4
gradient_accumulation_steps=2

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GLOBAL_BATCH_SIZE=$((per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS))


echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}🚀 Starting training with method: ${MAGENTA}$METHOD${NC}"
echo -e "${CYAN}Checkpoint name: ${BLUE}$CKPT${NC}"
echo -e "${CYAN}Alpha: ${BLUE}$ALPHA${NC}"
echo -e "${CYAN}Attention: ${BLUE}$ATTENTION${NC}"
echo -e "${CYAN}Global batch size: ${BLUE}$GLOBAL_BATCH_SIZE${NC}"
echo -e "${GREEN}=========================================${NC}"

deepspeed llava/train/train_mem_KD_TCD.py \
    --method $METHOD \
    --attn_implementation $ATTENTION \
    --deepspeed /root/wenzichen/EPIC_dev/scripts/zero3.json \
    --model_name_or_path /root/wenzichen/hf_models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /root/wenzichen/train_llava/LLaVA/playground/data/llava_v1_5_mix665k.json \
    --image_folder /root/wenzichen/train_llava/LLaVA/playground/data \
    --vision_tower /root/wenzichen/hf_models/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /root/wenzichen/hf_models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /root/wenzichen/EPIC_dev/checkpoints/${CKPT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --alpha $ALPHA \
    --disable_dropout False \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --sparse True \
    --pruned_layer 2 \
    --do_second_pruned False \
    --only_do_second_pruned False \
    --image_token_start_index 35 \
    --image_token_length 576 \
    --reduction_ratio 0.5 \
    --max_num_trunction 10000 \
    --pivot_image_token 4 \
    --pivot_text_token 4 \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --report_to tensorboard \
    --resume_from_checkpoint "/root/wenzichen/EPIC_dev/checkpoints/${CKPT}"

echo "Training completed for method: $METHOD"
echo "Checkpoint saved to: /root/wenzichen/EPIC_dev/checkpoints/${CKPT}"
