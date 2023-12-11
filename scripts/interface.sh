baichuan_belle=/data/nvme3/trained_ckpt/saved_models_3.5M/checkpoint-11270
baichuan_med_v4=/data/nvme3/trained_ckpt/saved_models_med_v4/checkpoint-1194
baichuan_med_v5=/public/home/tanyang/workspace/MedChatZH/model/baichuan-7b-med

CUDA_VISIBLE_DEVICES=0 python src/interface.py \
--model_name_or_path $baichuan_med_v5 \
--model_name baichuan \
--precision fp16 \
--port 17860
