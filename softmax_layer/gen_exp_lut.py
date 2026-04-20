#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from softmax_reference import EXP_WIDTH, build_exp_lut, write_mem


def main() -> None:
    out_path = Path(__file__).resolve().parent / "exp_lut.mem"
    lut = build_exp_lut()
    write_mem(out_path, lut, EXP_WIDTH)
    print(f"Wrote {len(lut)} entries to {out_path}")


if __name__ == "__main__":
    main()
