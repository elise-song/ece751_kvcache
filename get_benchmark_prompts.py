import os
from datasets import load_dataset


# downgrade the versions to be able to run
# pip install datasets==2.16.0
# pip install huggingface-hub==0.20.0
# pip install --upgrade huggingface_hub


def llama_3_instruct_prompt_format(prompt):
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return formatted_prompt

def longbench():
    OUTPUT_DIR = "benchmarks/LongBench/Meta-Llama-3-Instruct"

    task2prompt = {
        "2wikimqa": "Answer the question based on the given passages.\n\nThe following are given passages.\n{context}\n\nNow, answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.",
        "repobench-p": "Please complete the code given below. \n{context}{input}\n\nNow, provide the next line of code:\n"
    }

    tasks = ["2wikimqa", "gov_report", "multi_news", "repobench-p"]
    for task in tasks:
        dataset = load_dataset("THUDM/LongBench", task, split="test")

        print(f"Loaded {len(dataset)} samples")

        for i, sample in enumerate(dataset):
            prompt = task2prompt[task].format(**sample)
            prompt = llama_3_instruct_prompt_format(prompt)

            filename = os.path.join(OUTPUT_DIR, f"{task}_{i:04d}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(prompt)


if __name__ == '__main__':
    longbench()

