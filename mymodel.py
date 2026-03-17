import torch
from transformers import AutoTokenizer, LlamaForCausalLM, DynamicCache, ThinKCache

ratio = 0.4
recent_size=128

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.float16, device_map="auto")
f = open("benchmarks/LongBench/Meta-Llama-3-Instruct/2wikimqa_0000.txt", "r", encoding="utf-8")
prompt = f.read()
# prompt ="Large Language Models (LLMs) have revolutionized the field of natural language processing, achieving unprecedented performance across a variety of applications. However, their increased computational and memory demands present significant challenges, especially when handling long sequences. This paper focuses on the long-context scenario, addressing the inefficiencies in KV cache memory consumption during inference. Unlike existing approaches that optimize the memory based on the sequence length, we identify substantial redundancy in the channel dimension of the KV cache, as indicated by an uneven magnitude distribution and a low-rank structure in the attention weights. In response, we propose ThinK, a novel query-dependent KV cache pruning method designed to minimize attention weight loss while selectively pruning the least significant channels. Our approach not only maintains or enhances model accuracy but also achieves a reduction in KV cache memory costs by over 20% compared with vanilla KV cache eviction and quantization methods. What is this paper about? "
f.close()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_length = inputs.input_ids.shape[1]



# past_key_values = DynamicCache(config=model.config)
past_key_values = ThinKCache(
    config=model.config,
    recent_size=recent_size,  
    ratio=ratio,        
)

generated_ids = model.generate(**inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=1024)
print(tokenizer.decode(generated_ids[0][prompt_length:]))


# print(model.get_memory_footprint)


#  The expected shape for each tensor in the CacheLayers is 
# [batch_size, num_heads, seq_len, head_dim]

def think_savings(past_key_values):
    pruned_shape = past_key_values.layers[0].keys_pruned.shape
    recent_shape = past_key_values.layers[0].keys_recent.shape
    values_shape = past_key_values.layers[0].values.shape


    print("pruned keys: "+ str(pruned_shape))
    print("recent keys: "+ str(recent_shape))
    print("values: "+ str(values_shape))


    think_size = pruned_shape[2]*pruned_shape[3] + recent_shape[2]*recent_shape[3]
    value_size = values_shape[2]*values_shape[3]
    key_savings = think_size / value_size

    value_savings = value_size / value_size
    return (key_savings, value_savings)

print(think_savings(past_key_values))

    


