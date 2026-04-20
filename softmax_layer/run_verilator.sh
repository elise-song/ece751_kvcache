#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

module load verilator

VECTOR_LEN="${VECTOR_LEN:-8}"
NUM_CASES="${NUM_CASES:-3}"

python3 gen_exp_lut.py
python3 gen_test_vectors.py --vector-len "${VECTOR_LEN}" --num-cases "${NUM_CASES}"

verilator --binary --timing --sv -Wall -Wno-fatal \
  --top-module tb_softmax_layer \
  -GVECTOR_LEN="${VECTOR_LEN}" \
  -GNUM_CASES="${NUM_CASES}" \
  softmax_layer.sv \
  tb_softmax_layer.sv

./obj_dir/Vtb_softmax_layer
