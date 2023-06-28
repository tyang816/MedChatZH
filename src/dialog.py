import torch
import os, sys
import argparse
import time
sys.path.append(os.getcwd())
from src.models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from src.models.baichuan.modeling_baichuan import BaiChuanForCausalLM
from src.models.baichuan.configuration_baichuan import BaiChuanConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--precision', type=str, required=True)
args = parser.parse_args()



generation_config = dict(
    top_k=40, 
    top_p=0.75, 
    temperature=0.1, 
    repetition_penalty=1.2,
    num_beams=2, 
    max_new_tokens=512, 
    min_new_tokens=5,
    eos_token_id=2, 
    bos_token_id=1, 
    pad_token_id=0,
    return_dict_in_generate=True, 
    output_scores=False,
)



if __name__ == '__main__':
    device = torch.device('cuda', 0)
    
    if args.precision == 'fp16':
        load_type = torch.float16
    elif args.precision == 'fp32':
        load_type = torch.float32 #Sometimes may need torch.float32
        
    print("Loading model...")
    model_config = BaiChuanConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BaiChuanTokenizer.from_pretrained(args.model_name_or_path)
    model = BaiChuanForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=load_type, config=model_config
    ).to(device)
    model.eval()

    history = []
    print("ZhaoYan: 你好，我是兆言。")
    while True:
        print('-'*50)
        inputs = input("User: ")
        start_pos, end_pos = 0, 0
        final_res = ""
        for res in model.stream_chat(tokenizer=tokenizer, query=inputs, history=history, gen_kwargs=generation_config):
            end_pos = len(res)
            print(res[start_pos: end_pos], end="")
            start_pos = end_pos
            final_res = res
        history.append(f"Human: \n{inputs}\n\nAssistant: {final_res}\n")
        # limit history to 3
        if len(history) > 3:
            history = history[-3:]
        # prompt = ''.join(history)
        # inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # generation_output = model.generate(
        #     input_ids=input_ids, **generation_config
        # )
        # output = generation_output.sequences[0]
        
        # response = tokenizer.decode(output, skip_special_tokens=True).split("Assistant:")[-1].strip()
        # history[-1] += f"{response}\n"
