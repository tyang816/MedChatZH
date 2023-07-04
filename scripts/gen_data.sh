CUDA_VISIBLE_DEVICES=0 python src/gen_data_from_model.py \
--model_name_or_path /data/nvme3/checkpoint/baichuan-7B \
--model_name baichuan \
--precision fp16