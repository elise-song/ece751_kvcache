#!/usr/bin/env python3

from __future__ import annotations

import math
import struct
from pathlib import Path


DATA_WIDTH = 16
FRAC_BITS = 8
EXP_WIDTH = 16
OUT_WIDTH = 16
OUT_FRAC_BITS = 16
LUT_ADDR_WIDTH = 8
LUT_SIZE = 1 << LUT_ADDR_WIDTH
LUT_RANGE = 8.0
LUT_STEP = LUT_RANGE / LUT_SIZE


def to_hex(value: int, bits: int) -> str:
    return f"{value & ((1 << bits) - 1):0{bits // 4}x}"


def write_mem(path: Path, values: list[int], bits: int) -> None:
    lines = [to_hex(value, bits) for value in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def float_to_fp16_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">e", value), byteorder="big", signed=False)


def fp16_bits_to_float(bits: int) -> float:
    return struct.unpack(">e", int(bits).to_bytes(2, byteorder="big", signed=False))[0]


def clamp_signed(value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return max(lo, min(hi, value))


def build_exp_lut() -> list[int]:
    lut = []
    for idx in range(LUT_SIZE):
        x = idx * LUT_STEP
        y = math.exp(-x)
        q = int(round(y * ((1 << (EXP_WIDTH - 1)) - 1)))
        lut.append(max(0, min((1 << EXP_WIDTH) - 1, q)))
    return lut


def fp16_bits_to_q8_8(bits: int) -> int:
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF

    if exp == 0x1F:
        return clamp_signed(-0x8000 if sign else 0x7FFF, DATA_WIDTH)

    if exp == 0 and frac == 0:
        return 0

    if exp == 0:
        mant = frac
        shift = 1 - 15 - 10 + FRAC_BITS
    else:
        mant = (1 << 10) | frac
        shift = exp - 15 - 10 + FRAC_BITS

    if shift >= 0:
        magnitude = mant << shift
    else:
        rshift = -shift
        magnitude = (mant + (1 << (rshift - 1))) >> rshift

    if sign:
        magnitude = -magnitude

    return clamp_signed(magnitude, DATA_WIDTH)


def q0_16_to_fp16_bits(q: int) -> int:
    q = max(0, min((1 << OUT_WIDTH) - 1, int(q)))
    if q == 0:
        return 0

    if q < 4:
        frac = q << 8
        return frac & 0x03FF

    msb = q.bit_length() - 1
    exponent = msb - 1
    leading = 1 << msb
    remainder = q - leading

    if msb > 10:
        shift = msb - 10
        frac = (remainder + (1 << (shift - 1))) >> shift
    elif msb < 10:
        frac = remainder << (10 - msb)
    else:
        frac = remainder

    if frac >= (1 << 10):
        frac = 0
        exponent += 1

    exponent = max(0, min(0x1E, exponent))
    return ((exponent & 0x1F) << 10) | (frac & 0x03FF)


def reference_softmax_fp16(scores_fp16_bits: list[int], lut: list[int]) -> dict[str, list[int]]:
    scores_q8_8 = [fp16_bits_to_q8_8(bits) for bits in scores_fp16_bits]
    max_score = max(scores_q8_8)
    exp_values = []

    for score in scores_q8_8:
        diff = max_score - score
        if diff < 0:
            diff = 0
        if diff >= (8 << FRAC_BITS):
            lut_idx = LUT_SIZE - 1
        else:
            lut_idx = diff >> 3
        exp_values.append(lut[lut_idx])

    sum_exp = sum(exp_values)
    if sum_exp == 0:
        outputs_q0_16 = [0 for _ in scores_fp16_bits]
    else:
        outputs_q0_16 = []
        for exp_value in exp_values:
            normalized = ((exp_value << OUT_FRAC_BITS) + (sum_exp >> 1)) // sum_exp
            if normalized > (1 << OUT_WIDTH) - 1:
                normalized = (1 << OUT_WIDTH) - 1
            outputs_q0_16.append(normalized)

    outputs_fp16_bits = [q0_16_to_fp16_bits(value) for value in outputs_q0_16]

    return {
        "scores_q8_8": scores_q8_8,
        "exp_values_q1_15": exp_values,
        "outputs_q0_16": outputs_q0_16,
        "outputs_fp16_bits": outputs_fp16_bits,
    }
