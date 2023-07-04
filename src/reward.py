import argparse
import torch
from transformers import LlamaTokenizer
from transformers import AutoModelForSequenceClassification, LlamaForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    reward_model = reward_model.eval().half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, add_eos_token=True)
    prefix_user = "Human:"
    prefix_bot = "\n\nAssistant:"
    query = "重庆有哪些名胜古迹？"
    response = "1、涪陵榨菜产自重庆涪陵，是用青菜头腌制，配以辣椒和辅助香料等。在涪陵满山遍野到处可见到一种奇特的绿色或紫红色叶的蔬菜植物，当地人称之为“包包菜”、“疙瘩菜”或“青菜块”。吃起来酸脆爽口，吃面或者吃饭的时候都是一道很好下饭的菜，虽不是主菜，但是属于那种可以给很多菜系加分的配菜，吃上一片榨菜，让习惯浓油重辣的口中也可以多上许多不一样的清爽。2、羊角豆干武隆有句习语：羊角有三宝，豆干、老醋、猪腰枣。多次卤制的豆干，可以用香飘万里来形容。当时的羊角豆干，是乌江岸边纤夫的口粮，当时的羊角，几乎家家户户都做豆干，然后由妇女用提篮盛着，到乌江边叫卖。由于羊角豆干物美价廉，深受过往客商的喜欢，因此，做豆干买卖成了当地人谋生的主要手段。3、灯影牛肉灯影牛肉是从最开始的五香牛肉片衍生而来的，因为厚肉硬，吃时难嚼，且易塞牙，后来改进之后变成现在这样，牛肉又大又薄，烘烤过后还变得酥香可口。灯影也是指这种肉片的薄，薄到可以透过灯光，如同皮影戏的幕布一样，透过灯影来进行表演。"
    text = prefix_user+query+prefix_bot+response
    batch = tokenizer(text, return_tensors="pt",padding=True,truncation=True,max_length=1024)
    with torch.no_grad():
        reward = reward_model(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
        print(reward.item())