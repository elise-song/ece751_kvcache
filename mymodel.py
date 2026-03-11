import torch
from transformers import AutoTokenizer, LlamaForCausalLM, DynamicCache, ThinKCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("My sister loves jazz music but I like rock music because", return_tensors="pt").to(model.device)


past_key_values = DynamicCache(config=model.config)
# past_key_values = ThinKCache(
#     config=model.config,
#     recent_size=128,  # keep last 128 tokens at full precision
#     ratio=0.3,        # prune 30% of key channels
# )

outputs = model(**inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=200,)
# torch.set_printoptions(profile="full")
# print("keys shape: " + str(outputs.past_key_values.layers[0].keys.shape)) 
# print("values shape: " + str(outputs.past_key_values.layers[0].values.shape)) 
# print(outputs.past_key_values.layers[0].keys) 
# print(outputs.past_key_values.layers[0].values) 