import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import AutoModelForSequenceClassification, LlamaForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    reward_model = reward_model.eval().half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, add_eos_token=True)
    
    prefix_user = "Human:"
    prefix_bot = "\n\nAssistant:"
    
    lines = open(args.input_file, "r").readlines()
    lines_with_score = []
    
    for l in tqdm(lines):
        l = json.loads(l)
        query = l["question"]
        response = str(l["prediction"])
        text = prefix_user+query+prefix_bot+response
        batch = tokenizer(text, return_tensors="pt",padding=True,truncation=True,max_length=1024)
        with torch.no_grad():
            reward = reward_model(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
            l[args.model_name_or_path.split("/")[-1]] = reward.item()
            lines_with_score.append(l)
    
    with open(args.input_file.split(".")[0] + ".score.json", "w") as f:
        for l in lines_with_score:
            f.write(json.dumps(l, ensure_ascii=False)+"\n")