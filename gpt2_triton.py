#!/usr/bin/env python3
"""Provides the GEMM kernels GPT-2 small's layers compile from, a plain product and one activating its left operand.

A host issues each compiled image once per block of BLOCK_M token rows, the image walking every output column itself.
"""

from __future__ import annotations

import triton
import triton.language as tl

_TOKENS: int = 64
"""Token rows each kernel is planned at, a host issuing the image once per BLOCK_M of them."""

_MODEL: int = 768
"""GPT-2 small's model width, which every projection but the feed-forward down projection contracts over."""

_ROOT_TWO: tl.constexpr = tl.constexpr(1.4142135623730951)
"""Divisor the exact GELU scales its argument by, which Triton reaches from a kernel only as a constexpr."""


@triton.jit
def gpt2_matmul(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = _MODEL,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _MODEL,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _MODEL,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _TOKENS,
    N: tl.constexpr = _MODEL,
    K: tl.constexpr = _MODEL,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 256,
):
    """Multiplies the activations by the weights, carrying the accumulator across the whole contraction.

    Notes:
        BLOCK_K is the widest power of two dividing both of GPT-2's contractions, which keeps the number of
        contraction groups the image re-primes its staging for at its smallest.
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
            tl.store(output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gpt2_matmul_gelu(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    stride_am: tl.constexpr = 4 * _MODEL,
    stride_ak: tl.constexpr = 1,
    stride_bk: tl.constexpr = _MODEL,
    stride_bn: tl.constexpr = 1,
    stride_cm: tl.constexpr = _MODEL,
    stride_cn: tl.constexpr = 1,
    M: tl.constexpr = _TOKENS,
    N: tl.constexpr = _MODEL,
    K: tl.constexpr = 4 * _MODEL,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 256,
):
    """Multiplies the exact GELU of the activations by the weights, the activation riding each tile's staging."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            activation_pointers = activation_pointer + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            weight_pointers = weight_pointer + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                widened = tl.load(activation_pointers).to(tl.float32)
                activated = (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(tl.bfloat16)
                accumulator += tl.dot(activated, tl.load(weight_pointers))
                activation_pointers += BLOCK_K * stride_ak
                weight_pointers += BLOCK_K * stride_bk
            tl.store(output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)
