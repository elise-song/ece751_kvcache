git clone https://github.com/elise-song/ece751_kvcache.git
cd ece751_kvcache
git checkout think_h2o
pip install -e .
# python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
export PATH=$PATH:/home/esong32/.local/bin
export HF_TOKEN= <YOUR_HF_TOKEN>
hf auth login --token $HF_TOKEN
pip install accelerate
python3 scripts/LongBench/run_longbench.py --save_dir data/ --model_path meta-llama/Meta-Llama-3-8B-Instruct --method h2o --recent_size 512 --hh_ratio $1