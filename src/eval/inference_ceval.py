import torch
import json
from peft import PeftModel
import os, sys
sys.path.append(os.getcwd())
from models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from models.baichuan.modeling_baichuan import BaiChuanForCausalLM
from models.baichuan.configuration_baichuan import BaiChuanConfig
from transformers import GenerationConfig
import random 
import argparse
from tqdm import tqdm
#import gradio as gr
import json, os
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--precision', type=str, required=True)
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--few_shot', action="store_true")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
def generate_prompt(input_text):
    return "Human: \n" + input_text + "\n\nAssistant:\n"


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=8,
    min_new_tokens=1,
    repetition_penalty=1.2,
    **kwargs,
):
#    prompt = generate_prompt(input)
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = tokenizer.decode(output, skip_special_tokens=True).split("Assistant:")[-1].strip()

        yield output

if __name__ == '__main__':
    if args.precision == 'fp16':
        load_type = torch.float16
    elif args.precision == 'fp32':
        load_type = torch.float32 #Sometimes may need torch.float32
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    tokenizer = BaiChuanTokenizer.from_pretrained(args.model_name_or_path)
    model_config = BaiChuanConfig.from_pretrained(args.model_name_or_path)

    print(f"Loading model {args.model_name_or_path}...")
    model = BaiChuanForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=load_type, config=model_config,device_map="auto"
    )

    if device==torch.device('cpu'):
        model.float()

#    model.to(device)
    model.eval()
    print("Load model successfully")

    tasks=  [e[:-9]  for e in  os.listdir('ceval')]


    task2desc = {
        "high_school_physics": "高中物理",
        "fire_engineer": "注册消防工程师",
        "computer_network": "计算机网络",
        "advanced_mathematics": "高等数学",
        "logic": "逻辑学",
        "middle_school_physics": "初中物理",
        "clinical_medicine": "临床医学",
        "probability_and_statistics": "概率统计",
        "ideological_and_moral_cultivation": "思想道德修养与法律基础",
        "operating_system": "操作系统",
        "middle_school_mathematics": "初中数学",
        "chinese_language_and_literature": "中国语言文学",
        "electrical_engineer": "注册电气工程师",
        "business_administration": "工商管理",
        "high_school_geography": "高中地理",
        "modern_chinese_history": "近代史纲要",
        "legal_professional": "法律职业资格",
        "middle_school_geography": "初中地理",
        "middle_school_chemistry": "初中化学",
        "high_school_biology": "高中生物",
        "high_school_chemistry": "高中化学",
        "physician": "医师资格",
        "high_school_chinese": "高中语文",
        "tax_accountant": "税务师",
        "high_school_history": "高中历史",
        "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
        "high_school_mathematics": "高中数学",
        "professional_tour_guide": "导游资格",
        "veterinary_medicine": "兽医学",
        "environmental_impact_assessment_engineer": "环境影响评价工程师",
        "basic_medicine": "基础医学",
        "education_science": "教育学",
        "urban_and_rural_planner": "注册城乡规划师",
        "middle_school_biology": "初中生物",
        "plant_protection": "植物保护",
        "middle_school_history": "初中历史",
        "high_school_politics": "高中政治",
        "metrology_engineer": "注册计量师",
        "art_studies": "艺术学",
        "college_economics": "大学经济学",
        "college_chemistry": "大学化学",
        "law": "法学",
        "sports_science": "体育学",
        "civil_servant": "公务员",
        "college_programming": "大学编程",
        "middle_school_politics": "初中政治",
        "teacher_qualification": "教师资格",
        "computer_architecture": "计算机组成",
        "college_physics": "大学物理",
        "discrete_mathematics": "离散数学",
        "marxism": "马克思主义基本原理",
        "accountant": "注册会计师",
    }

    res={}
    log_data={}
    i=0
    #succes=[]
    for task in tasks:
        data=[]
        data=pd.read_csv(f'ceval/{task}_test.csv')
        task_name=task

        res[task_name]={}
        log_data[task_name]={}

        if i>0:break
        

        prompt=pd.read_csv(f'dev/{task}_dev.csv')
        prefixs=[]
        for line in prompt.values:
            part1="Human: \n"+f"以下是中国关于{task2desc[task_name]}考试的单项选择题,请在A、B、C、D、4个选项中选择正确的一个，填充在问题的____部分中。\n问题：\n"
            part2=line[1]
            part3="选项：\n"+"A:"+str(line[2])+"\nB:"+str(line[3])+"\nC:"+str(line[4])+"\nD:"+str(line[5])
            part4="\n\nAssistant:\n答案是："+line[6]+'\n'
            temp_instruction=part1+part2+part3+part4
            prefixs.append(temp_instruction)

        for line in tqdm(data.values) :
            log_data[task_name][str(line[0])]={}
            # if args.multi:
            #     if line[0]%8 != args.device:
            #         continue


            part1="Human: \n"+f"以下是中国关于{task2desc[task_name]}考试的单项选择题,请在A、B、C、D、4个选项中选择正确的一个，填充在问题的____部分中。\n问题：\n"
            part2=line[1]
            part3="选项：\n"+"A:"+str(line[2])+"\nB:"+str(line[3])+"\nC:"+str(line[4])+"\nD:"+str(line[5])
            part4="\n\nAssistant:\n答案是："

            instruction=part1+part2+part3+part4

            log_data[task_name][str(line[0])]["original_instruct"]=instruction            
            # for temp in prefixs:
            #     instruction=temp+instruction
            # instruction=instruction[:-1*max_new_tokens]                

            #inputs = tokenizer(instruction, max_length=max_new_tokens,truncation=True,return_tensors="pt")

            if args.few_shot:
                for temp in prefixs:
                    temp_instruction=temp+instruction
                    temp_inputs = tokenizer(temp_instruction, return_tensors="pt")
                    if temp_inputs["input_ids"].shape[1]>max_new_tokens:
                        break 
                    else:
                        instruction=temp+instruction
                
            gen=evaluate(instruction)
            answer=next(gen)


            log_data[task_name][str(line[0])]["generate_text"]=instruction  
            log_data[task_name][str(line[0])]["answer"]=answer  


            answer_1=[]
            if "A" in answer or str(line[2]) in answer:
                answer_1.append("A")
            if "B" in answer or str(line[3]) in answer:
                answer_1.append("B")               
            if "C" in answer or str(line[4]) in answer:
                answer_1.append("C")
            if "D" in answer or str(line[5]) in answer:
                answer_1.append("D")
            if len(answer_1)>1:
                ans=random.choice(answer_1)
            elif len(answer_1)==0:
                ans=random.choice(["A","B","C","D"])
            else:
                ans=answer_1[0]

            res[task_name][str(line[0])]=ans
            log_data[task_name][str(line[0])]["ans"]=ans

        i+=1    

    with open(f'ceval_result/ceval_result_0621_{str(args.device)}.json','w',encoding='utf-8') as f:
        json.dump(res,f,ensure_ascii=False,indent=4)

    with open(f'ceval_result/ceval_result_0621_{str(args.device)}_log.json','w',encoding='utf-8') as f:
        json.dump(log_data,f,ensure_ascii=False,indent=4)
