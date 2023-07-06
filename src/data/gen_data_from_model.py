import torch
import argparse
import os, sys
import pandas as pd
import gradio as gr
import json, os
sys.path.append(os.getcwd())
from peft import PeftModel
from tqdm import tqdm
from src.models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from src.models.baichuan.modeling_baichuan import BaiChuanForCausalLM
from src.models.baichuan.configuration_baichuan import BaiChuanConfig
from src.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from src.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from src.models.chatglm.configuration_chatglm import ChatGLMConfig
from transformers import GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--precision', type=str, required=True)
parser.add_argument('--gen_num', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--output_dir', type=str, default='gen_data')
args = parser.parse_args()


generation_config = dict(
    temperature=1,
    top_k=64000,
    top_p=1,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1,
    max_new_tokens=2048
)


if __name__ == '__main__':
    if args.precision == 'fp16':
        load_type = torch.float16
    elif args.precision == 'fp32':
        load_type = torch.float32 #Sometimes may need torch.float32
        
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    
    print(f"Loading model {args.model_name_or_path}...")
    if args.model_name == 'baichuan':
        tokenizer = BaiChuanTokenizer.from_pretrained(args.model_name_or_path)
        model_config = BaiChuanConfig.from_pretrained(args.model_name_or_path)
        model = BaiChuanForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
    elif args.model_name == 'llama':        
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        model_config = LlamaConfig.from_pretrained(args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
    elif args.model_name == 'chatglm':
        tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)
        model_config = ChatGLMConfig.from_pretrained(args.model_name_or_path)
        model = ChatGLMForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
        
    if device==torch.device('cpu'):
        model.float()

    model.to(device)
    model.eval()
    print("Load model successfully")
    
    data = {"text": [], "id": []}
    idx = 0
    
    for _ in tqdm(range(args.gen_num // args.batch_size)):
        inputs = torch.tensor([[1] for _ in range(args.batch_size)])
        generation_output = model.generate(
            input_ids = inputs.to(device), 
            **generation_config
        )

        for output in generation_output:
            generate_text = tokenizer.decode(output, skip_special_tokens=True)
            data["text"].append(generate_text)
            data["id"].append(idx+1)
            idx += 1
    
    data = pd.DataFrame(data)
    data.to_csv(f"{args.output_dir}/{args.model_name}_{args.precision}_6.csv", index=False)
    