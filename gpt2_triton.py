#!/usr/bin/env python3
"""Provides the kernels GPT-2 small's layers compile from, being its GEMMs, LayerNorms, attention and adds.

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

_EPSILON: tl.constexpr = tl.constexpr(1e-5)
"""Variance floor GPT-2's LayerNorm adds before taking the reciprocal square root."""

_HEAD: int = 64
"""Width of one of GPT-2 small's attention heads."""

_MASKED: tl.constexpr = tl.constexpr(-1.0e30)
"""Score a key after its query takes, which no score reaches, so its probability is exactly zero."""


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


@triton.jit
def gpt2_add(
    left_pointer: "*bf16",
    right_pointer: "*bf16",
    output_pointer: "*bf16",
    M: tl.constexpr = _TOKENS,
    N: tl.constexpr = _MODEL,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 256,
):
    """Adds two [tokens, features] tensors element by element, being a block's residual add."""
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
            tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) + tl.load(right_pointer + offsets))


@triton.jit
def gpt2_add_bias(
    left_pointer: "*bf16",
    bias_pointer: "*bf16",
    output_pointer: "*bf16",
    M: tl.constexpr = _TOKENS,
    N: tl.constexpr = _MODEL,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 256,
):
    """Adds one bias row to every token row of a [tokens, features] tensor, being a projection's bias."""
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            offsets = offs_m[:, None] * N + offs_n[None, :]
            bias = tl.load(bias_pointer + offs_n)
            tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) + bias[None, :])


@triton.jit
def gpt2_layernorm(
    input_pointer: "*bf16",
    gamma_pointer: "*bf16",
    beta_pointer: "*bf16",
    output_pointer: "*bf16",
    M: tl.constexpr = _TOKENS,
    N: tl.constexpr = _MODEL,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 256,
):
    """Normalises each token row of a [tokens, features] tensor, then scales and shifts it by gamma and beta."""
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for tile in range(0, N, BLOCK_N):
            offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
            total += tl.load(input_pointer + offsets).to(tl.float32)
        mean = tl.sum(total, 1) / N
        squares = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for tile in range(0, N, BLOCK_N):
            offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
            centered = tl.load(input_pointer + offsets).to(tl.float32) - mean[:, None]
            squares += centered * centered
        scale = tl.rsqrt(tl.sum(squares, 1) / N + _EPSILON)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            offsets = offs_m[:, None] * N + offs_n[None, :]
            centered = tl.load(input_pointer + offsets).to(tl.float32) - mean[:, None]
            gamma = tl.load(gamma_pointer + offs_n).to(tl.float32)
            beta = tl.load(beta_pointer + offs_n).to(tl.float32)
            normalised = centered * scale[:, None] * gamma[None, :] + beta[None, :]
            tl.store(output_pointer + offsets, normalised.to(tl.bfloat16))


@triton.jit
def gpt2_attention(
    query_pointer: "*bf16",
    key_pointer: "*bf16",
    value_pointer: "*bf16",
    output_pointer: "*bf16",
    stride_m: tl.constexpr = _HEAD,
    M: tl.constexpr = _TOKENS,
    D: tl.constexpr = _HEAD,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
):
    """Attends one head causally with the online softmax, its 1/sqrt(head) already folded into the queries."""
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(query_pointer + offs_m[:, None] * stride_m + offs_d[None, :])
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, M, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(key_pointer + offs_n[:, None] * stride_m + offs_d[None, :])
            value = tl.load(value_pointer + offs_n[:, None] * stride_m + offs_d[None, :])
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], tl.dot(query, tl.trans(key)), _MASKED)
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        normalised = accumulator / running_sum[:, None]
        tl.store(output_pointer + offs_m[:, None] * stride_m + offs_d[None, :], normalised.to(tl.bfloat16))
