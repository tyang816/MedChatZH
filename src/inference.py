import torch
import json
import os, sys
sys.path.append(os.getcwd())
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.models.tokenization_baichuan import BaiChuanTokenizer
from src.models.modeling_baichuan import BaiChuanForCausalLM
from src.models.configuration_baichuan import BaiChuanConfig
from peft import PeftModel
import argparse
from tqdm import tqdm
import json, os
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
args = parser.parse_args()


max_new_tokens = 1024
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)

instruction_list = [
    "Human: \n小明有12个橙子，他想把它们分给他的4个朋友，每人分到的橙子数量相同，每人能分到几个橙子？\n\nAssistant:\n",
    "Human: \n以下是一道小学数学题：小明家里有 3 只宠物猫和 2 只宠物狗，小花家里有 4 只宠物猫和 1 只宠物狗，谁家里宠物更多？\n\nAssistant:\n",
    "Human: \n题目：小明有5个球，他送给小红2个球，还剩多少个球？\n\nAssistant:\n",
    "Human: \n请问2+3等于几？\n\nAssistant:\n"
]


if __name__ == '__main__':
    load_type = torch.float16 #Sometimes may need torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    
    tokenizer = BaiChuanTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = BaiChuanConfig.from_pretrained(args.model_name_or_path)

    if args.use_lora:
        base_model = BaiChuanForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
        model = PeftModel.from_pretrained(base_model, args.ckpt_path, device_map='auto')
    else:
        model = BaiChuanForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=load_type, config=model_config)

    if device==torch.device('cpu'):
        model.float()

    #model.to(device)
    model.eval()
    print("Load model successfully")
    history = ""
    for instruction in instruction_list:
        inputs = tokenizer(instruction, max_length=max_new_tokens,truncation=True,return_tensors="pt")
        generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device), 
            **generation_config
        )[0]

        generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
        print(generate_text)
        print("-"*100)
