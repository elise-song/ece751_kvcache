#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from softmax_reference import (
    DATA_WIDTH,
    OUT_WIDTH,
    build_exp_lut,
    float_to_fp16_bits,
    fp16_bits_to_float,
    reference_softmax_fp16,
    write_mem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-len", type=int, default=8)
    parser.add_argument("--num-cases", type=int, default=3)
    return parser.parse_args()


def build_case(case_id: int, vector_len: int) -> list[float]:
    if case_id == 0:
        return [0.0 for _ in range(vector_len)]

    if case_id == 1:
        # A long repeated ramp across the useful LUT range.
        return [((idx % 17) - 8) * 0.5 for idx in range(vector_len)]

    if case_id == 2:
        # One strong peak followed by a broad tail and repeated offsets.
        values = []
        for idx in range(vector_len):
            if idx == vector_len // 3:
                values.append(6.0)
            elif idx == (2 * vector_len) // 3:
                values.append(4.0)
            else:
                values.append(((idx % 29) - 14) * 0.25)
        return values

    # Additional cases use a deterministic pseudo-waveform without randomness.
    return [((idx * (case_id + 3)) % 41 - 20) * 0.2 for idx in range(vector_len)]


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    lut = build_exp_lut()

    cases_float = [build_case(case_id, args.vector_len) for case_id in range(args.num_cases)]
    cases_fp16 = [[float_to_fp16_bits(value) for value in case] for case in cases_float]
    references = [reference_softmax_fp16(case, lut) for case in cases_fp16]

    flat_inputs = [item for case in cases_fp16 for item in case]
    flat_expected = [item for ref in references for item in ref["outputs_fp16_bits"]]

    write_mem(base / "tb_inputs.mem", flat_inputs, DATA_WIDTH)
    write_mem(base / "tb_expected.mem", flat_expected, OUT_WIDTH)

    manifest = {
        "vector_len": args.vector_len,
        "num_cases": args.num_cases,
        "cases_float": cases_float,
        "cases_fp16_hex": [[f"0x{value:04x}" for value in case] for case in cases_fp16],
        "decoded_inputs_float": [[fp16_bits_to_float(value) for value in case] for case in cases_fp16],
        "scores_q8_8": [ref["scores_q8_8"] for ref in references],
        "exp_values_q1_15": [ref["exp_values_q1_15"] for ref in references],
        "expected_q0_16": [ref["outputs_q0_16"] for ref in references],
        "expected_fp16_hex": [[f"0x{value:04x}" for value in ref["outputs_fp16_bits"]] for ref in references],
    }
    (base / "tb_vectors.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases_float)} test cases of length {args.vector_len} to {base}")


if __name__ == "__main__":
    main()
