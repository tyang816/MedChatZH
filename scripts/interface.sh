CUDA_VISIBLE_DEVICES=0 python src/interface.py \
--model_name_or_path /data/nvme3/trained_ckpt/saved_models_med_v4/checkpoint-1194 \
--model_name baichuan \
--precision fp16 \
--port 17860
