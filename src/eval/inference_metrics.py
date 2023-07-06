import torch
import argparse
import json
import datasets
import peft
import os, sys
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import  PeftModel
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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--precision', type=str, required=True)
parser.add_argument('--use_lora', action='store_true')
parser.add_argument('--eval_dataset', type=str, required=True)
args = parser.parse_args()


generation_config = dict(
    top_k=40, 
    top_p=0.75, 
    temperature=0.1, 
    repetition_penalty=1.2,
    num_beams=1, 
    max_new_tokens=512, 
    min_new_tokens=5,
    eos_token_id=2, 
    bos_token_id=1, 
    pad_token_id=0,
    return_dict_in_generate=True, 
    output_scores=False,
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
        ).half()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
    
    if args.use_lora:
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    
    if device==torch.device('cpu'):
        model.float()

    model.to(device)
    model.eval()
    print("Load model successfully")
    
    data = load_dataset("json", data_files=args.eval_dataset)
    final_result = {'label': [], 'predict': []}
    
    if args.model_name == 'chatglm':
        for label, question in tqdm(zip(data['train']['label'], data['train']['question'])):
            if label in final_result['label']:
                continue
            print(question)
            inputs = tokenizer([question], return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model.generate(**inputs, **generation_config)
            outputs = outputs[0][0]
            response = tokenizer.decode(outputs, skip_special_tokens=True)[len(question):].strip()
            final_result['label'].append(label)
            print(response)
            final_result['predict'].append(response)
            
    else:
        def tokenize_fn(examples):
            sentence = "Human: \n" + examples["question"] + "\n\nAssistant: \n"
            examples = tokenizer(sentence, max_length=1024, truncation=True)
            return examples
   
        column_names = data["train"].column_names
        test_data = data["train"].map(
            tokenize_fn,
            remove_columns=column_names
        )
        
        def collate_fn(batch):
            batch = tokenizer.pad(batch, padding=True, return_tensors="pt")
            return batch
        test_data_loader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)
        
        for idx, batch in tqdm(enumerate(test_data_loader)):
            label = data["train"]["label"][idx]
            if label in final_result['label']:
                continue
            final_result['label'].append(label)
            generation_output = model.generate(
                input_ids = batch['input_ids'].to(device),
                attention_mask = batch['attention_mask'].to(device),
                **generation_config
            )[0]

            for output in generation_output:
                generate_text = tokenizer.decode(output, skip_special_tokens=True).split("Assistant: \n")[1].strip()
                final_result['predict'].append(generate_text)
    
    final_result = pd.DataFrame(final_result)
    final_result.to_csv(f"{args.model_name}_valid.csv", index=False)    

    