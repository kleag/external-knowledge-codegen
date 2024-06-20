import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import transformers
import torch

# CHECKPOINT = "/home/data/dataset/huggingface/LLMs/deepseek-ai/deepseek-coder-6.7b-instruct"
DEVICE = "cuda"
INPUT_PATHS = ['data/conala/conala-train.json','data/conala/conala-test.json']
OUTPUT_PATH="data/conala/llm_output/"
PRE_PROMPT="Generate the Python code \n"


pipeline = transformers.pipeline(
    model="/home/data/dataset/huggingface/LLMs/bigcode/starcoder2-15b-instruct-v0.1",
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def respond(instruction: str, response_prefix: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False)
    prompt += response_prefix

    teminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("###"),
    ]

    result = pipeline(
        prompt,
        max_length=256,
        num_return_sequences=1,
        do_sample=False,
        eos_token_id=teminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        truncation=True,
    )
    response = response_prefix + result[0]["generated_text"][len(prompt) :].split("###")[0].rstrip()
    return response

instruction = "Write a function in Python for : "
response_prefix = ""

print(respond(instruction, response_prefix))


if __name__ == "__main__":
    for input_path in INPUT_PATHS:
        output_file = f"{OUTPUT_PATH}{os.path.splitext(os.path.basename(input_path))[0]}.jsonl"
        print("output "+output_file, flush=True)
        with open(input_path, 'r') as input_file, open(output_file, 'w') as output_file:
            data = json.load(input_file)
            for i, line in enumerate(data):
                print(f"-> {i}/{len(data)} ----------------------------", flush=True)
                print("line", line, flush=True)
                
                if line['rewritten_intent'] is None:
                    intent = line['intent']
                else:
                    intent = line['rewritten_intent']
                    
                code = respond(instruction+intent.strip(), "")
                output_file.write(json.dumps({"text": instruction+intent.strip(), "code": code}) + '\n')
                output_file.flush()  # Flush the output file after each write
                # print("elapsed_time", elapsed_time, flush=True)
        
        print(f"Code generation f{input_path} completed!", flush=True)