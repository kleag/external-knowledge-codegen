import json
import os
import re
import sys
import time
import transformers
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
INPUT_PATHS = [
    'data/conala/conala-test.json',
    'data/conala/conala-train.json'
    ]
OUTPUT_PATH = "data/conala/llm_snippet_output"
# PRE_PROMPT = "Generate the Python code \n"
# PRE_PROMPT = "Write a function in Python for : "
PRE_PROMPT = "Write a Python one liner for: "

# MODEL = "/home/data/dataset/huggingface/LLMs/deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL = "/home/data/dataset/huggingface/LLMs/bigcode/starcoder2-15b-instruct-v0.1"


def respond(instruction: str, response_prefix: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = pipeline.tokenizer.apply_chat_template(messages,
                                                    tokenize=False)
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
    response = (response_prefix
                + result[0]["generated_text"][len(prompt) :].split(
                    "###")[0].rstrip())
    response = re.search(r'```(.*?)```', response, re.DOTALL).group(1)
    match = re.search(r'^python\n(.*)\n$', response, re.DOTALL)
    if match:
        response = match.group(1)
    match = re.search(r'^(import.*; )?(.*)', response, re.DOTALL)
    if match:
        response = match.group(2)
    return response



if __name__ == "__main__":
    pipeline = transformers.pipeline(
        model=MODEL,
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    instruction = PRE_PROMPT
    response_prefix = ""
    # print(f"Test call to respond with instruction `{instruction}` "
    #     f"and prefix `{response_prefix}`:",
    #     file=sys.stderr)
    # print(respond(instruction, response_prefix), file=sys.stderr)
    for input_path in INPUT_PATHS:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        output_file = os.path.join(
            OUTPUT_PATH,
            f"{os.path.splitext(os.path.basename(input_path))[0]}.jsonl")
        print(f"output {output_file}", file=sys.stderr)
        with (open(input_path, 'r') as input_file,
              open(output_file, 'w') as output_file):
            data = json.load(input_file)
            for i, line in enumerate(tqdm(data)):
                # print(f"-> {i}/{len(data)} ----------------------------",
                #       file=sys.stderr)
                # print("line", line, file=sys.stderr)
                
                if line['rewritten_intent'] is None:
                    intent = line['intent']
                else:
                    intent = line['rewritten_intent']
                    
                code = respond(instruction+intent.strip(), "")
                output_file.write(json.dumps(
                    {"text": instruction+intent.strip(),
                     "code": code}) + '\n')
                output_file.flush()  # Flush the output file after each write
        
        print(f"Code generation f{input_path} completed!", file=sys.stderr)
