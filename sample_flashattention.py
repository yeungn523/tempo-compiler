#!/usr/bin/env python3
"""Provides the sample flash attention kernel this compiler is exercised against."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import triton
import triton.language as tl

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "board_package"))

_QUERIES: int = 32
"""Queries the kernel attends with, walked in BLOCK_M steps, being the token rows a host issues a block each."""

_KEYS: int = 64
"""Keys the kernel attends over, walked in BLOCK_N steps, being the loop one image runs itself."""

_HEAD: int = 64
"""Width of one attention head, being GPT-2 small's model width over its heads."""

_SEED: int = 0
"""Draw the operands are generated from, fixed so two runs compile the same numbers."""

_SPREAD: float = 2.0
"""Bound the operands are drawn inside, which keeps every score under the range the SFU's table covers."""


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
    """Writes the operands the kernel above attends over, at the extents it declares.

    Returns:
        Zero once every file is written.
    """
    written = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
    written.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(_SEED)
    shapes = (("query.npy", (_QUERIES, _HEAD)), ("key.npy", (_KEYS, _HEAD)), ("value.npy", (_KEYS, _HEAD)))
    for name, shape in shapes:
        operand = generator.uniform(-_SPREAD, _SPREAD, size=shape).astype(np.float32)
        np.save(written / name, operand)
        print(f"wrote {written / name}  {operand.shape} {operand.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
