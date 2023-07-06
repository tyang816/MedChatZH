import torch
import json
import argparse
import mdtex2html
import gradio as gr
import os, sys
sys.path.append(os.getcwd())
from peft import PeftModel

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
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def generate_prompt(input_text):
    return "Human: \n" + input_text + "\n\nAssistant:\n"


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=512,
    min_new_tokens=5,
    repetition_penalty=1.2,
    **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
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
            output_scores=False
            )
        output = generation_output.sequences[0]
        output = tokenizer.decode(output, skip_special_tokens=True).split("Assistant:")[1].strip()
        print(output)
        yield output


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
    if args.model_name != "chatglm":
        # https://gradio.app/docs/
        gr.Interface(
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2, label="Input", placeholder="Welcome to the zhaoyan model"
                ),
                gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
                gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
                gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
                gr.components.Slider(minimum=1, maximum=4, step=1, value=1, label="Beams Number"),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=10, value=512, label="Max New Tokens"
                ),
                gr.components.Slider(
                    minimum=1, maximum=300, step=10, value=5, label="Min New Tokens"
                ),
                gr.components.Slider(
                    minimum=1.0, maximum=2.0, step=0.1, value=1.2, label="Repetition Penalty"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=25,
                    label="Output",
                )
            ],
            title="兆言大模型",
        ).queue().launch(share=True, server_name="0.0.0.0", server_port=args.port)
    else:
        with gr.Blocks() as demo:
            gr.HTML("""<h1 align="center">ChatGLM</h1>""")

            chatbot = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                            container=False)
                    with gr.Column(min_width=32, scale=1):
                        submitBtn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=1):
                    emptyBtn = gr.Button("Clear History")
                    max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

            history = gr.State([])

            submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                            show_progress=True)
            submitBtn.click(reset_user_input, [], [user_input])

            emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

        demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=args.port)
