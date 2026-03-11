import torch
from transformers import AutoTokenizer, LlamaForCausalLM, DynamicCache, ThinKCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("Large Language Models (LLMs) have revolutionized the field of natural language processing, achieving unprecedented performance across a variety of applications. However, their increased computational and memory demands present significant challenges, especially when handling long sequences. This paper focuses on the long-context scenario, addressing the inefficiencies in KV cache memory consumption during inference. Unlike existing approaches that optimize the memory based on the sequence length, we identify substantial redundancy in the channel dimension of the KV cache, as indicated by an uneven magnitude distribution and a low-rank structure in the attention weights. In response, we propose ThinK, a novel query-dependent KV cache pruning method designed to minimize attention weight loss while selectively pruning the least significant channels. Our approach not only maintains or enhances model accuracy but also achieves a reduction in KV cache memory costs by over 20% compared with vanilla KV cache eviction and quantization methods. What is this paper about? ",
    # "Large language models (LLMs) (Hadi et al., 2023; Brown et al., 2020; OpenAI, 2023; Touvron et al., 2023a; b; Scao et al., 2022; Reid et al., 2024) have emerged as a dominant paradigm in natural language processing, achieving state-of-the-art performance across various tasks. A key principle, the Scaling Law (Kaplan et al., 2020), suggests that LLMs exhibit emergent abilities as model size increases, improving their capacity to understand complex context and handle long sequences (Xiong et al., 2023). This growth in capacity enables LLMs to generate coherent, contextually accurate responses and supports a variety of downstream applications, such as document summarization (Zhang et al., 2019; 2024a), code generation (Chen et al., 2021b), solving mathematical problems (Hendrycks et al., 2021; Zhou et al., 2023; Wang et al., 2023; Lightman et al., 2023), and conversational AI (OpenAI, 2022; 2023).",
    # "Large language models (LLMs) (Hadi et al., 2023; Brown et al., 2020; OpenAI, 2023; Touvron et al., 2023a; b; Scao et al., 2022; Reid et al., 2024) have emerged as a dominant paradigm in natural language processing, achieving state-of-the-art performance across various tasks. A key principle, the Scaling Law (Kaplan et al., 2020), suggests that LLMs exhibit emergent abilities as model size increases, improving their capacity to understand complex context and handle long sequences (Xiong et al., 2023). This growth in capacity enables LLMs to generate coherent, contextually accurate responses and supports a variety of downstream applications, such as document summarization (Zhang et al., 2019; 2024a), code generation (Chen et al., 2021b), solving mathematical problems (Hendrycks et al., 2021; Zhou et al., 2023; Wang et al., 2023; Lightman et al., 2023), and conversational AI (OpenAI, 2022; 2023). Despite their success in various applications, generating outputs with LLMs incurs significant computational and financial costs, which rise with increasing model size and sequence length. Both the training (Strubell et al., 2020; Hoffmann et al., 2022; Dong et al., 2024a) and inference (Ainslie et al., 2023) stages involve frequent generation, further contributing to these costs. Consequently, efficient LLMs have gained traction in recent years (Hu et al., 2021; Wan et al., 2023). To address these challenges, quantization (Frantar et al., 2022; Lin et al., 2024; Dettmers et al., 2024; Xu et al., 2023) and pruning methods (Sun et al., 2023; Frantar & Alistarh, 2023; Lu et al., 2024b) are employed to reduce model size. Additionally, the key-value (KV) cache, stored in GPU memory alongside model parameters, scales linearly with both sequence length and batch size, creating a substantial memory burden when handling long sequences. Consequently, effective management of extended contexts is essential for the practical deployment of LLMs. In this paper, we focus on the long-context scenario, aiming to reduce memory consumption associated with processing lengthy sequences.", 
                   return_tensors="pt").to(model.device)

prompt_length = inputs.input_ids.shape[1]

# past_key_values = DynamicCache(config=model.config)
past_key_values = ThinKCache(
    config=model.config,
    recent_size=32,  # keep last 32 tokens at full precision
    ratio=0.5,        # prune 30% of key channels
)

# outputs = model(**inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=200,)
# print(outputs.)
generated_ids = model.generate(**inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=20,)
print(tokenizer.batch_decode(generated_ids)[0])

# torch.set_printoptions(profile="full")
# print("keys shape: " + str(outputs.past_key_values.layers[0].keys_pruned.shape)) 
# print("keys shape: " + str(outputs.past_key_values.layers[0].keys_recent.shape)) 

# print("values shape: " + str(outputs.past_key_values.layers[0].values.shape)) 
# print(outputs.past_key_values.layers[0].keys) 
# print(outputs.past_key_values.layers[0].values) 