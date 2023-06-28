#! /bin/bash

model_name_or_path=/data/nvme3/trained_ckpt/saved_models_3.5M/checkpoint-11270 # or bloomz-7b1-mt

train_file=/home/tyang/cq_info_conv.json
validation_file=/home/tyang/cq_info_conv.json
output_dir=/data/nvme3/trained_ckpt/saved_models_cqinfo
mkdir -p ${output_dir}

cache_dir=/data/nvme3/cache_dir/hf_cache_dir_cqinfo
mkdir -p ${cache_dir}
cutoff_len=1024


# #FT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --model_name baichuan \
    --deepspeed src/configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 5 \
    --num_train_epochs 5 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
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