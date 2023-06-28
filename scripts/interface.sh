CUDA_VISIBLE_DEVICES=7 python src/interface.py \
--model_name_or_path /data/nvme2/trained_llama_ckpt/saved_models_cqinfo/checkpoint-40 \
--model_name llama \
--precision fp16 \
--port 17862
