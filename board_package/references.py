"""Provides the host references the device paths are held to, each rounding where the device rounds."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The board package's directory."""

sys.path.insert(0, str(HERE))

from bf16 import (  # noqa: E402
    _BLOCK_DEPTH,
    _LN_EPSILON,
    _MODEL_WIDTH,
    _on_bf16_grid,
    as_bf16_words,
    from_bf16_words,
    gelu_lut_reference,
)


def layernorm_reference(
    rows: NDArray[np.float32], gamma: NDArray[np.float32], beta: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Returns the LayerNorm a BF16 device computes, on the host.

    Notes:
        Rounds where the device rounds: the mean and variance are BF16 sums scaled by a BF16 1/width,
        and the affine is two BF16 operations. What is left is the rsqrt table's interpolation.
    """
    values = _on_bf16_grid(values=rows)
    inverse = _on_bf16_grid(values=np.float32(1.0 / _MODEL_WIDTH))
    mean = _on_bf16_grid(values=values.sum(axis=1, keepdims=True, dtype=np.float32) * inverse)
    centred = _on_bf16_grid(values=values - mean)
    variance = _on_bf16_grid(
        values=_on_bf16_grid(values=centred * centred).sum(axis=1, keepdims=True, dtype=np.float32)
        * inverse)
    scale = _on_bf16_grid(values=1.0 / np.sqrt(_on_bf16_grid(values=variance + _LN_EPSILON)))
    return _on_bf16_grid(values=_on_bf16_grid(values=centred * scale) * _on_bf16_grid(values=gamma)
                         + _on_bf16_grid(values=beta))


def softmax_reference(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns the row-wise softmax a BF16 device computes, on the host.

    Notes:
        Quantises at each step the device quantises at: the subtraction, the exponential and the
        product all round to BF16, and the reciprocal is the SFU's own. What is left over is the
        table's interpolation error, which is what a check's bound has to cover.

    Args:
        scores: One row per reduction, each `_VREG_ELEMENTS` wide.

    Returns:
        The float32 softmax of each row.
    """
    values = _on_bf16_grid(values=scores)
    shifted = _on_bf16_grid(values=values - values.max(axis=1, keepdims=True))
    weights = _on_bf16_grid(values=np.exp(shifted, dtype=np.float32))
    totals = _on_bf16_grid(values=weights.sum(axis=1, keepdims=True, dtype=np.float32))
    return _on_bf16_grid(values=weights * _on_bf16_grid(values=1.0 / totals))


def transformer_block_reference(hidden: NDArray[np.float32], weights: dict) -> NDArray[np.float32]:
    """Returns the block output a BF16 device computes, on the host."""
    normalised = layernorm_reference(rows=hidden, gamma=weights["ln_1.g"], beta=weights["ln_1.b"])
    fused = _on_bf16_grid(values=_device_product(normalised, weights["attn.qkv.w"]) + weights["attn.qkv.b"])
    query, key, value = np.split(fused, 3, axis=1)
    heads = 12
    head = hidden.shape[1] // heads
    gathered = np.concatenate(
        [attention_reference(query=np.ascontiguousarray(query[:, i * head:(i + 1) * head]),
                             key=np.ascontiguousarray(key[:, i * head:(i + 1) * head]),
                             value=np.ascontiguousarray(value[:, i * head:(i + 1) * head]), causal=True)
         for i in range(heads)], axis=1)
    attended = _on_bf16_grid(values=_device_product(_on_bf16_grid(values=gathered), weights["attn.proj.w"])
                             + weights["attn.proj.b"])
    residual = _on_bf16_grid(values=hidden + attended)

    normalised = layernorm_reference(rows=residual, gamma=weights["ln_2.g"], beta=weights["ln_2.b"])
    expanded = _on_bf16_grid(values=_device_product(normalised, weights["mlp.fc.w"]) + weights["mlp.fc.b"])
    activated = from_bf16_words(words=gelu_lut_reference(words=as_bf16_words(values=expanded)))
    contracted = _on_bf16_grid(values=_device_product(activated, weights["mlp.proj.w"]) + weights["mlp.proj.b"])
    return _on_bf16_grid(values=residual + contracted)


def attention_reference(
    query: NDArray[np.float32], key: NDArray[np.float32], value: NDArray[np.float32], *, causal: bool = False
) -> NDArray[np.float32]:
    """Returns the attention output a BF16 device computes, on the host.

    Args:
        query: [tokens, head] queries.
        key: [tokens, head] keys.
        value: [tokens, head] values.
        causal: Determines whether a query attends only at or before its own position, every later key's score
            replaced by the large negative constant the device's mask adds.

    Returns:
        The [tokens, head] attention output.
    """
    head = query.shape[1]
    scaled = _on_bf16_grid(values=(query / math.sqrt(head)).astype(np.float32))
    scores = _on_bf16_grid(values=(scaled.astype(np.float64) @ _on_bf16_grid(values=key).T
                                   .astype(np.float64)).astype(np.float32))
    if causal:
        later = np.arange(key.shape[0])[None, :] > np.arange(query.shape[0])[:, None]
        scores = np.where(later, np.float32(-1.0e30), scores)
    weights = softmax_reference(scores=scores)
    return (weights.astype(np.float64) @ _on_bf16_grid(values=value).astype(np.float64)).astype(np.float32)


def _device_product(left: NDArray[np.float32], right: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns a product summed as a compiled GEMM sums it, rounding each block's partial and the running sum to BF16.

    Notes:
        The array sums one `_BLOCK_DEPTH` block of the contraction exactly, and the image adds that partial into a
        BF16 accumulator. Rounding only the finished product misses the error this adds where an output is a small
        difference of large partials, which GPT-2's largest activations are.

    Args:
        left: [rows, depth] left operand, on the BF16 grid.
        right: [depth, columns] right operand, on the BF16 grid.

    Returns:
        The [rows, columns] product.
    """
    total = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for start in range(0, left.shape[1], _BLOCK_DEPTH):
        block = slice(start, start + _BLOCK_DEPTH)
        partial = _on_bf16_grid(values=(left[:, block].astype(np.float64) @ right[block].astype(np.float64))
                                .astype(np.float32))
        total = _on_bf16_grid(values=total + partial)
    return total
