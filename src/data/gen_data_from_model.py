import torch
import json
from peft import PeftModel
import os, sys
sys.path.append(os.getcwd())
from src.models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from src.models.baichuan.modeling_baichuan import BaiChuanForCausalLM
from src.models.baichuan.configuration_baichuan import BaiChuanConfig
from src.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from src.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from src.models.chatglm.configuration_chatglm import ChatGLMConfig
from transformers import GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import mdtex2html

import argparse
from tqdm import tqdm
import gradio as gr
import json, os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--precision', type=str, required=True)
args = parser.parse_args()


generation_config = dict(
    temperature=1,
    top_k=100,
    top_p=1,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.5,
    max_new_tokens=1024
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
    
    for _ in range(10):
        inputs = torch.tensor([[1]])
    
        print(inputs)
        generation_output = model.generate(
            input_ids = inputs.to(device), 
            **generation_config
        )[0]

        generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
        print(generate_text)