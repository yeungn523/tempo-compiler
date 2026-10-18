#!/usr/bin/env python3
"""Provides the sample GEMM and flash attention kernels this compiler is exercised against."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import triton
import triton.language as tl

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
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

_QUERIES: int = 32
"""Queries the attention kernel attends with, walked in BLOCK_M steps, being the token rows a host issues a block each."""

_KEYS: int = 64
"""Keys the attention kernel attends over, walked in BLOCK_N steps, being the loop one image runs itself."""

_HEAD: int = 64
"""Width of one attention head, being GPT-2 small's model width over its heads."""

_SPREAD: float = 1.25
"""Bound the operands are drawn inside, which keeps the scores of one query row within 64 of one another."""

_SEED: int = 0
"""Draw the operands are generated from, fixed so two runs compile the same numbers."""


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
    BLOCK_N: tl.constexpr = 64,
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
def flash_attention_kernel(
    query_pointer: "*bf16",
    key_pointer: "*bf16",
    value_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_qm: tl.constexpr = _HEAD,
    stride_qd: tl.constexpr = 1,
    stride_kn: tl.constexpr = _HEAD,
    stride_kd: tl.constexpr = 1,
    stride_vn: tl.constexpr = _HEAD,
    stride_vd: tl.constexpr = 1,
    stride_om: tl.constexpr = _HEAD,
    stride_od: tl.constexpr = 1,
    M: tl.constexpr = _QUERIES,
    N: tl.constexpr = _KEYS,
    D: tl.constexpr = _HEAD,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
):
    """Attends with the online softmax flash attention carries, one key block at a time.

    Notes:
        The running maximum, the running sum and the accumulator cross every trip of the key loop, so the scores
        of one block are exponentiated and consumed where they lie and never reach memory.
    """
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(query_pointer + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(key_pointer + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            value = tl.load(value_pointer + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            scores = tl.dot(query, tl.trans(key))
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(output_pointer + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 accumulator / running_sum[:, None])


def main() -> int:
    """Writes the operands the kernels above run on, at the widest extents any of them declares.

    Returns:
        Zero once every file is written.
    """
    written = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    written.mkdir(parents=True, exist_ok=True)
    left, right = build_operands(_ROWS, _DEPTH, _COLUMNS, kind="varying", seed=_SEED)
    generator = np.random.default_rng(_SEED)
    query, key, value = (generator.uniform(-_SPREAD, _SPREAD, size=shape).astype(np.float32)
                         for shape in ((_QUERIES, _HEAD), (_KEYS, _HEAD), (_KEYS, _HEAD)))
    for name, operand in (("activations.npy", left), ("weights.npy", right), ("query.npy", query),
                          ("key.npy", key), ("value.npy", value)):
        np.save(written / name, operand)
        print(f"wrote {written / name}  {operand.shape} {operand.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
