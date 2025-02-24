import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


run_key = True
while run_key:
    print("Введите запрос. Для выхода из кода нажмите end")
    data = sys.stdin.read()
    if data == "end":
        break
    messages = [
        {"role": "system", "content": "you are an assistant for converting data into rdf format Turtle; rdf Turtle should be as accurate as possible regarding the sent text"},
        {"role": "user", "content": data}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1000)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(result)
