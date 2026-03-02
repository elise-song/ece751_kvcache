import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

past_key_values = DynamicCache(config=model.config)
outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
print(outputs.past_key_values.layers[0].keys) 
print(outputs.past_key_values.layers[0].values) 