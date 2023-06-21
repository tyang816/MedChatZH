import torch
import os, sys
sys.path.append(os.getcwd())
from src.models.tokenization_baichuan import BaiChuanTokenizer
from src.models.modeling_baichuan import BaiChuanForCausalLM
from src.models.configuration_baichuan import BaiChuanConfig

ckpt = '/home/tyang/baichuan/saved_models/checkpoint-9015'
device = torch.device('cuda', 0)
model_config = BaiChuanConfig.from_pretrained(ckpt)

print("Loading model...")
model = BaiChuanForCausalLM.from_pretrained(ckpt, config=model_config, device_map='auto')
tokenizer = BaiChuanTokenizer.from_pretrained(ckpt)
model.eval()
history = ''
while True:
    print('-'*200)
    inputs = input("用户: ")
    prompt = history + f"Human: \n{inputs}\n\nAssistant: \n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True, top_k=30, top_p=0.85, temperature=0.5, repetition_penalty=1.2, eos_token_id=2, bos_token_id=1, pad_token_id=0)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = output.split("Assistant: \n")[-1]
    history = prompt + response
    print(f'系统：{response}')
    print('-' * 200)