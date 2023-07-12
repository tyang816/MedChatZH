baichuan_med=/data/nvme3/trained_ckpt/saved_models_med_v4/checkpoint-1194
baichuan=/data/nvme3/trained_ckpt/saved_models_3.5M/checkpoint-11270
lora_llama_med=/home/tyang/lora-llama-med
chatglm_med=/data/nvme3/checkpoint/chatglm-6b-med

CUDA_VISIBLE_DEVICES=1 python src/eval/inference.py \
--model_name_or_path $baichuan \
--model_name baichuan-belle \
--precision fp16 \
--eval_dataset data/eval/webMedQA/webMedQA.jsonl