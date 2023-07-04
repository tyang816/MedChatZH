#! /bin/bash

model_name_or_path=/home/tyang/BELLE-LLaMA-EXT-13B # or bloomz-7b1-mt

train_file=/data/nvme3/data/med_single_v4_train_conv.json
validation_file=/data/nvme3/data/med_single_v4_valid_conv.json
output_dir=/data/nvme2/trained_llama_ckpt/saved_models_med_v4
mkdir -p ${output_dir}

cache_dir=/data/nvme2/cache_dir/hf_cache_dir_med_v4
mkdir -p ${cache_dir}
cutoff_len=1024


# #FT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_sf.py \
    --model_name_or_path ${model_name_or_path} \
    --model_name llama \
    --deepspeed src/configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 20 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --eval_steps 300 \
    --learning_rate 1e-4 \
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