# Script for evaluating KV cache pruning techniques on LongBench
# Derived from: https://github.com/SalesforceAIResearch/ThinK/blob/main/ThinK_eager/run_longbench.py

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, H2OCache

task2prompt = {
    "2wikimqa": "Answer the question based on the given passages.\n\nThe following are given passages.\n{context}\n\nNow, answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.",
    "repobench-p": "Please complete the code given below. \n{context}{input}\n\nNow, provide the next line of code:\n"
}

def format_prompt(prompt, model_name):
    # Llama3
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return formatted_prompt

def post_process(response, model_name):
    # Llama3
    end_of_prompt =  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = response.strip()
    if response.find(end_of_prompt) > 0:
        response = response[:response.find(end_of_prompt)]
    return response

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def eval_dataset(args, model, tokenizer):
    print("Loading data...")

    test_data = []
    prompts = []
    inputs = []
    contexts = []
    answers = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []

    input_max_len = 0
    model_path = args.model_path.lower()

    with open(args.data_file, 'r') as f:
        for line in f:
            sample = json.loads(line)

            length = sample["length"]
            input_max_len = max(input_max_len, length)

            # Format prompt
            prompt = task2prompt[args.dataset].format(**sample)
            prompt = format_prompt(prompt, model_path)

            sample["prompt"] = prompt
            test_data.append(sample)

    print(f"Max length is {input_max_len}")

    for sample in test_data:
        prompts.append(sample["prompt"])
        inputs.append(sample["input"])
        contexts.append(sample["context"])
        answers.append(sample["answers"])
        lengths.append(sample["length"])
        datasets.append(sample["dataset"])
        languages.append(sample["language"])
        all_classess.append(sample["all_classes"])
        _ids.append(sample["_id"])

    print("Finished loading model and tokenizer")

    model_name = model_path.split("/")[-1]

    cache_info = "_"
    if args.method == "h2o":
        if args.hh_size is not None:
            hh = f"h2s{args.hh_size}"
        elif args.hh_ratio is not None:
            hh = f"h2r{args.hh_ratio}"
        else:
            hh = "h2r0.1"
        recent = f"recent{args.recent_size}"
        cache_info = f"{hh}_{recent}_"

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.method}_{cache_info}", args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.method}_{cache_info}", args.dataset, f"{args.method}.json"), "w")

    assert(args.eval_batch_size == 1)

    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answers[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]

        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]

        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True
        ).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids

        prompt_length = batch_input_ids.shape[-1]
        batch_prompt_lengths = [prompt_length]

        past_key_values = None
        if args.method != "default":
            if args.method == "h2o":
                if args.hh_size is not None:
                    hh_size = args.hh_size
                elif args.hh_ratio is not None:
                    hh_size = int(prompt_length * args.hh_ratio)
                else:
                    hh_size = int(prompt_length * 0.1)
                recent_size = args.recent_size
                past_key_values = H2OCache(config=model.config, hh_size=hh_size, recent_size=recent_size)

        try:
            output = model.generate(
                **tokenized_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                min_length=prompt_length+1,
                past_key_values=past_key_values
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"INFO: Skipping prompt {i}: CUDA out of memory")
            else:
                print(f"Error processing prompt {i}: {e}")
            continue
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            continue

        batch_outputs = tokenizer.batch_decode([output[0][prompt_length:]], skip_special_tokens=True)
        batch_n_generated = [len(output[0]) - prompt_length]

        if args.method == "h2o":
            batch_kv_seq_len = [past_key_values.get_seq_length()]
        else:
            batch_kv_seq_len = [len(output[0])]

        torch.cuda.empty_cache()

        for j in range(1):
            sample = {}
            sample["prompt"] = batch_prompts[j]
            sample["input"] = batch_inputs[j]
            sample["context"] = batch_contexts[j]
            sample["answers"] = batch_answerss[j]
            sample["pred"] = post_process(batch_outputs[j], model_path)
            sample["length"] = batch_lengths[j]

            sample["dataset"] = batch_datasets[j]
            sample["language"] = batch_languages[j]
            sample["all_classes"] = batch_all_classess[j]
            sample["_id"] = batch__ids[j]

            sample["prompt_length"] = batch_prompt_lengths[j]
            sample["n_generated"] = batch_n_generated[j]
            sample["kv_cache_seq_length"] = batch_kv_seq_len[j]

            fout.write(json.dumps(sample) + "\n")

    fout.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--recent_size", type=int, default=32, help="Number of recent tokens to keep")
    parser.add_argument("--hh_size", type=int, default=None, help="Number of HH tokens to keep")
    parser.add_argument("--hh_ratio", type=float, default=None, help="Number of HH tokens to keep as fraction of prompt size")
    args = parser.parse_args()

    set_seed(args.seed)

    # Set up model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype="auto", device_map="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    if args.method is None:
        args.method = "default"
    elif args.method.lower() in ["h2o"]:
        args.method = args.method.lower()
    else:
        print(f"WARN: {args.method} is not a valid KV cache method, defaulting to default cache")
        args.method = "default"

    datasets = [
        "2wikimqa",
        "multi_news",
        "gov_report",
        "repobench-p"
    ]

    if args.dataset is not None:
        if args.dataset in datasets:
            datasets = [args.dataset]
        else:
            raise ValueError(f"{args.dataset} is not a valid dataset")

    for i, dataset in enumerate(datasets):
        print(f"Working on dataset {dataset} - {i+1}/{len(datasets)}")
        args.dataset = dataset
        args.data_file = f"benchmarks/LongBench/{args.dataset}.jsonl"
        eval_dataset(args, model, tokenizer)


if __name__ == '__main__':
    main()
