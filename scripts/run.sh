#! /bin/bash

model_name_or_path=/home/tyang/checkpoint/baichuan-7B # or bloomz-7b1-mt

train_file=/home/tyang/data/belle-3.5M/train_3.5M_CN.json
validation_file=/home/tyang/data/medical/valid_0.5K_conv.json
output_dir=saved_models
output_dir_lora=saved_models_lora
mkdir -p ${output_dir}
mkdir -p ${output_dir_lora}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024


# #FT
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node 6 src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --deepspeed src/configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ./logs \
    --report_to "tensorboard"



