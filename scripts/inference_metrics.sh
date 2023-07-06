CUDA_VISIBLE_DEVICES=1 python src/eval/inference_metrics.py \
--model_name_or_path /data/nvme3/trained_ckpt/saved_models_med_v4/checkpoint-1194 \
--model_name baichuan \
--precision fp16 \
--eval_dataset data/eval/webMedQA/valid.jsonl