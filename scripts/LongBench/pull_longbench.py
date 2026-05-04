from datasets import load_dataset

datasets = ["2wikimqa", "gov_report", "multi_news", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
