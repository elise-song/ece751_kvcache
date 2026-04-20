# Softmax Layer

This folder contains a parameterized fp16-interface softmax block for the project diagram stage:

`score SRAM -> softmax -> score SRAM`

## Design Choice

The module is implemented as a framed streaming engine:

1. Load one vector of attention scores into local storage while tracking the max score.
2. Replay the stored scores, subtract the max, and map the result through an `exp(-x)` LUT.
3. Accumulate the exponential sum.
4. Replay the exponentials and output normalized softmax probabilities.

This is a practical hardware shape for the diagram because the upstream systolic array and SRAM already suggest a buffered, vector-based flow rather than a purely combinational softmax.

## Numeric Format

- Input score: IEEE-754 half precision (`fp16`, 16-bit)
- Exponential LUT output: unsigned `Q1.15` (`EXP_WIDTH=16`)
- Softmax output: IEEE-754 half precision (`fp16`, 16-bit)

Internally, the module converts fp16 inputs into signed `Q8.8`, performs a safe-softmax style `max` subtraction plus LUT-based exponentiation, normalizes in unsigned `Q0.16`, and converts the result back to fp16.

The LUT covers `exp(-x)` for `x in [0, 8)` with 256 entries. Values below `-8` after max subtraction are clamped to the smallest LUT entry.

## Files

- `softmax_layer.sv`: RTL module
- `tb_softmax_layer.sv`: Verilator testbench
- `softmax_reference.py`: small software model used to generate LUTs and expected outputs
- `gen_exp_lut.py`: emits `exp_lut.mem`
- `gen_test_vectors.py`: emits `tb_inputs.mem`, `tb_expected.mem`, and `tb_vectors.json`
- `run_verilator.sh`: builds and runs the Verilator simulation

## Run

```bash
cd /disk/zli2793/projects/ece751-prj/softmax_layer
./run_verilator.sh
```

For a full-length run that matches the intended SRAM-sized vector:

```bash
cd /disk/zli2793/projects/ece751-prj/softmax_layer
VECTOR_LEN=4096 NUM_CASES=3 ./run_verilator.sh
```

## Notes

- The RTL is parameterized, but the bundled testbench uses `VECTOR_LEN=8` to keep simulation fast.
- For the project datapath, you would typically instantiate the module with `VECTOR_LEN=4096`.
- The current normalization uses integer division in the output pass. That is straightforward and Verilator-friendly, though you may later replace it with a reciprocal/LUT pipeline if you want a more synthesis-focused implementation.
