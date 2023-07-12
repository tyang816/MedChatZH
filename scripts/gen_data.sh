CUDA_VISIBLE_DEVICES=7 python src/data/gen_data_from_model.py \
--model_name_or_path /data/nvme3/checkpoint/baichuan-7B \
--model_name baichuan \
--precision fp16 \
--batch_size 10 \
--gen_num 10000 \
--output_dir gen_data