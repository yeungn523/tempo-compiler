#!/usr/bin/env python3
"""Holds the GEMM kernel this compiler is exercised against, and writes the operand pair it runs on.

Compiling the kernel needs only the declaration below, which `kernel_entry_point.py` reads by name. Running it
needs numbers too, so this file writes an operand pair when it is executed rather than imported, into the directory
named as its one argument. The pair comes from `bf16.build_operands`, which refuses a draw whose rows or columns
repeat, because such a draw lets an image that reorders them look correct.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import triton
import triton.language as tl

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "board_package"))

from bf16 import build_operands  # noqa: E402

_ROWS: int = 64
"""Token rows the whole product covers, which the operands are drawn at and the kernel walks in BLOCK_M steps."""

_COLUMNS: int = 48
"""Output columns the whole product covers, walked in BLOCK_N steps."""

_DEPTH: int = 192
"""The contraction both operands share, walked in BLOCK_K steps."""

_SEED: int = 0
"""Draw the operands are generated from, fixed so two runs compile the same numbers."""


@triton.jit
def matmul_kernel(
    activation_pointer,
    weight_pointer,
    output_pointer,
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _COLUMNS,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _COLUMNS,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    N: tl.constexpr = _COLUMNS,
    K: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Multiplies an M by K activation by a K by N weight, one BLOCK_M by BLOCK_N output tile at a time.

    M, N and K are the extents of the whole product, and the three BLOCK extents are what one iteration of each
    loop covers, so their ratios are the trip counts the planner reports.

    The strides are declared here rather than passed at the launch, so each operand states the layout it lies in
    and the compiler reads it where the kernel wrote it. Both operands lie row major, which is what a stride of
    one along the lane axis says.
    """
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            activation_pointers = activation_pointer + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            weight_pointers = weight_pointer + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(activation_pointers), tl.load(weight_pointers))
                activation_pointers += BLOCK_K * stride_ak
                weight_pointers += BLOCK_K * stride_bk
            output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(output_pointers, accumulator)


def main() -> int:
    """Writes the operand pair the kernel above runs on, at the extents it declares.

    The directory is taken from the one argument this accepts, defaulting to the working directory.

    Returns:
        Zero once both files are written.
    """
    written = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    written.mkdir(parents=True, exist_ok=True)
    left, right = build_operands(_ROWS, _DEPTH, _COLUMNS, kind="varying", seed=_SEED)
    for name, operand in (("activations.npy", left), ("weights.npy", right)):
        np.save(written / name, operand)
        print(f"wrote {written / name}  {operand.shape} {operand.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
