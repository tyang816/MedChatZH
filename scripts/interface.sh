CUDA_VISIBLE_DEVICES=1 python src/interface.py \
--model_name_or_path /data/nvme2/trained_llama_ckpt/saved_models_med_v4/checkpoint-1194 \
--model_name llama \
--precision fp16 \
--port 17861
