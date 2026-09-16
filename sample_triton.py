#!/usr/bin/env python3
"""Provides the sample GEMM kernels and a full FFN this compiler is exercised against."""

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
"""Token rows every kernel covers, which the operands are drawn at and the kernels walk in BLOCK_M steps. This is
what the register file stages at once, which a kernel walking its token blocks inside the reduction holds live."""

_COLUMNS: int = 3072
"""Output columns the whole product covers, walked in BLOCK_N steps, being GPT-2's hidden width."""

_DEPTH: int = 768
"""The contraction both operands share, walked in BLOCK_K steps."""

_RESIDENT_COLUMNS: int = 96
"""Output columns a kernel walking its column tiles inside the reduction covers, being what one image holds
live at this depth."""

_RESIDENT_SQUARE_COLUMNS: int = 48
"""Output columns a kernel walking both output axes inside its reduction covers, which shares one block with every
token row the product holds."""

_SEED: int = 0
"""Draw the operands are generated from, fixed so two runs compile the same numbers."""

_PROJECTION_SCALE: float = 0.02
"""Deviation the feed-forward projections are drawn at, being the one GPT-2 initialises its own weights to. A
layer contracts twice and activates between, so weights drawn at unit deviation leave an intermediate two
orders of magnitude above anything a trained layer holds, which the BF16 chain carries far less accurately."""

_ROOT_TWO: tl.constexpr = tl.constexpr(1.4142135623730951)
"""Divisor the exact GELU scales its argument by before the error function reads it. Triton reaches a global from
a decorated function only where the value is a constexpr instance, so the annotation alone would not carry it."""


@triton.jit
def matmul_kernel(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _COLUMNS,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _COLUMNS,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    N: tl.constexpr = _COLUMNS,
    K: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 2048,
    BLOCK_N: tl.constexpr = 256,
    BLOCK_K: tl.constexpr = 64,
):
    """Multiplies an M by K activation by a K by N weight, walking the tokens, the columns, then the reduction.

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


@triton.jit
def matmul_n_m_k(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
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
    """Walks the columns, the tokens, then the reduction, which holds one column block's weights across the
    token blocks a host issues against it."""
    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(0, N, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        for block in range(0, M, BLOCK_M):
            offs_m = block + tl.arange(0, BLOCK_M)
            activation_pointers = activation_pointer + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            weight_pointers = weight_pointer + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(activation_pointers), tl.load(weight_pointers))
                activation_pointers += BLOCK_K * stride_ak
                weight_pointers += BLOCK_K * stride_bk
            output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(output_pointers, accumulator)


@triton.jit
def matmul_m_k_n(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _COLUMNS,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _COLUMNS,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    N: tl.constexpr = _RESIDENT_COLUMNS,
    K: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Walks the tokens, the reduction, then the columns, which keeps every column tile live to the end of the
    contraction and accumulates each through memory."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for depth in range(0, K, BLOCK_K):
            for tile in range(0, N, BLOCK_N):
                offs_n = tile + tl.arange(0, BLOCK_N)
                activation_pointers = (activation_pointer + offs_m[:, None] * stride_am
                                       + (depth + offs_k)[None, :] * stride_ak)
                weight_pointers = (weight_pointer + (depth + offs_k)[:, None] * stride_bk
                                   + offs_n[None, :] * stride_bn)
                output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(output_pointers, tl.load(output_pointers)
                         + tl.dot(tl.load(activation_pointers), tl.load(weight_pointers)))


@triton.jit
def matmul_n_k_m(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
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
    """Walks the columns, the reduction, then the tokens, which keeps every token block live to the end of the
    contraction and accumulates each through memory."""
    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(0, N, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        for depth in range(0, K, BLOCK_K):
            for block in range(0, M, BLOCK_M):
                offs_m = block + tl.arange(0, BLOCK_M)
                activation_pointers = (activation_pointer + offs_m[:, None] * stride_am
                                       + (depth + offs_k)[None, :] * stride_ak)
                weight_pointers = (weight_pointer + (depth + offs_k)[:, None] * stride_bk
                                   + offs_n[None, :] * stride_bn)
                output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(output_pointers, tl.load(output_pointers)
                         + tl.dot(tl.load(activation_pointers), tl.load(weight_pointers)))


@triton.jit
def matmul_k_m_n(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _COLUMNS,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _COLUMNS,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    N: tl.constexpr = _RESIDENT_SQUARE_COLUMNS,
    K: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Walks the reduction, the tokens, then the columns, which keeps the whole block live to the end of the
    contraction and accumulates it through memory."""
    offs_k = tl.arange(0, BLOCK_K)
    for depth in range(0, K, BLOCK_K):
        for block in range(0, M, BLOCK_M):
            offs_m = block + tl.arange(0, BLOCK_M)
            for tile in range(0, N, BLOCK_N):
                offs_n = tile + tl.arange(0, BLOCK_N)
                activation_pointers = (activation_pointer + offs_m[:, None] * stride_am
                                       + (depth + offs_k)[None, :] * stride_ak)
                weight_pointers = (weight_pointer + (depth + offs_k)[:, None] * stride_bk
                                   + offs_n[None, :] * stride_bn)
                output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(output_pointers, tl.load(output_pointers)
                         + tl.dot(tl.load(activation_pointers), tl.load(weight_pointers)))


@triton.jit
def matmul_k_n_m(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _COLUMNS,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _COLUMNS,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    N: tl.constexpr = _RESIDENT_SQUARE_COLUMNS,
    K: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Walks the reduction, the columns, then the tokens, which keeps the whole block live to the end of the
    contraction and accumulates it through memory."""
    offs_k = tl.arange(0, BLOCK_K)
    for depth in range(0, K, BLOCK_K):
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            for block in range(0, M, BLOCK_M):
                offs_m = block + tl.arange(0, BLOCK_M)
                activation_pointers = (activation_pointer + offs_m[:, None] * stride_am
                                       + (depth + offs_k)[None, :] * stride_ak)
                weight_pointers = (weight_pointer + (depth + offs_k)[:, None] * stride_bk
                                   + offs_n[None, :] * stride_bn)
                output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(output_pointers, tl.load(output_pointers)
                         + tl.dot(tl.load(activation_pointers), tl.load(weight_pointers)))


@triton.jit
def gelu(value):
    """Returns the exact GELU of one tile, written in the error function form the compiler fuses.
    """
    widened = value.to(tl.float32)
    return (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(value.dtype)


@triton.jit
def fused_ffn_kernel(
    activation_pointer: "*bf16",
    up_weight_pointer: "*bf16",
    down_weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _DEPTH,
    stride_ak: tl.constexpr = 1,
    stride_uk: tl.constexpr = _COLUMNS,
    stride_un: tl.constexpr = 1,
    stride_dk: tl.constexpr = _DEPTH,
    stride_dn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _DEPTH,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _ROWS,
    K: tl.constexpr = _DEPTH,
    H: tl.constexpr = _COLUMNS,
    N: tl.constexpr = _DEPTH,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 64,
    BLOCK_H: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
):
    """Runs both projections of a feed-forward layer and the activation between them, in one kernel.

    Notes:
        Each hidden chunk is activated and read by the down projection in the iteration that produced it, so the
        intermediate stays in registers.

        The output accumulates through memory across the hidden width, so the buffer is entered zeroed.
    """
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for chunk in range(0, H, BLOCK_H):
            activation_pointers = activation_pointer + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            up_pointers = up_weight_pointer + offs_k[:, None] * stride_uk + (chunk + offs_h)[None, :] * stride_un
            hidden = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                hidden += tl.dot(tl.load(activation_pointers), tl.load(up_pointers))
                activation_pointers += BLOCK_K * stride_ak
                up_pointers += BLOCK_K * stride_uk
            # Rounds the chunk to the width the array stages before the activation reads it.
            activated = gelu(hidden.to(tl.bfloat16))
            for tile in range(0, N, BLOCK_N):
                offs_n = tile + tl.arange(0, BLOCK_N)
                down_pointers = (down_weight_pointer + (chunk + offs_h)[:, None] * stride_dk
                                 + offs_n[None, :] * stride_dn)
                output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(output_pointers, tl.load(output_pointers) + tl.dot(activated, tl.load(down_pointers)))


def main() -> int:
    """Writes the operands the kernels above run on, at the widest extents any of them declares.

    The plain products read an activation and a weight, and the feed-forward layer a third operand besides,
    its two projections being contracted one after the other.

    Returns:
        Zero once every file is written.
    """
    written = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    written.mkdir(parents=True, exist_ok=True)
    left, right = build_operands(_ROWS, _DEPTH, _COLUMNS, kind="varying", seed=_SEED)
    # The down projection contracts the hidden width back to the model width, which is what the layer drains.
    _, down = build_operands(_ROWS, _COLUMNS, _DEPTH, kind="varying", seed=_SEED + 1)
    projections = [(operand * _PROJECTION_SCALE).astype(np.float32) for operand in (right, down)]
    for name, operand in (("activations.npy", left), ("weights.npy", right),
                          ("up_weight.npy", projections[0]), ("down_weight.npy", projections[1])):
        np.save(written / name, operand)
        print(f"wrote {written / name}  {operand.shape} {operand.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
