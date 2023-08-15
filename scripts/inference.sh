baichuan_med=/data/nvme3/trained_ckpt/saved_models_med_v4/checkpoint-1194
baichuan=/data/nvme3/trained_ckpt/saved_models_3.5M/checkpoint-11270
lora_llama_med=/data/nvme2/checkpoint/lora-llama-med
chatglm_med=/data/nvme2/checkpoint/chatglm-6b-med

CUDA_VISIBLE_DEVICES=3 python src/eval/inference.py \
--model_name_or_path $chatglm_med \
--model_name chatglm \
--precision fp16 \
--eval_dataset data/eval/cMedQA2/cMedQA2.jsonl