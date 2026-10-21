#!/usr/bin/env python3
"""Provides the RTL check that the attention images the MLIR pipeline compiles from GPT-2's kernels attend every head
of a sequence, or of a decode step, in one launch.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The directory holding the RTL checks."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "board_package"))

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _ZERO_TILE_WORDS,
    as_bf16_words,
    attention_bf16_reference,
    attention_reference,
    decode_mask_tiles,
    from_bf16_words,
    pack_attention_constants,
)
from driver import compile_kernel  # noqa: E402
from gpt2_triton import gpt2_attention, gpt2_decode_attention  # noqa: E402
from rtl import run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the head is allowed past what the BF16 model costs over the same operands."""

_HEAD: int = 64
"""Width of one attention head, being GPT-2 small's."""

_SPREAD: float = 1.25
"""Bound the operands are drawn inside, which keeps the scores inside the range the exponential table covers."""

_MASKED: float = -1.0e30
"""Score a masked key takes in the references, matching what the image's mask tiles add."""


def simulate(*, tokens: int, heads: int, build_dir: str | None = None,
             max_cycles: int | None = None) -> tuple[float, float]:
    """Runs every head of every token block against the RTL in one launch and returns what it and a BF16 model cost.

    Args:
        tokens: Tokens the sequence holds, a whole number of token blocks.
        heads: Heads the fused tensor carries.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles the round may take, or None for the bound the testbench carries.

    Returns:
        The worst head's error against the exact causal reference, and the worst error the BF16 model carries.
    """
    compiled = compile_kernel(gpt2_attention, heads=heads, M=tokens)
    generator = np.random.default_rng(0)
    query, key, value = (_drawn(generator, rows=tokens, heads=heads) for _ in range(3))
    fused = _pack(np.concatenate([query, key, value], axis=1))
    dumped = run_image(compiled.image, arguments=[compiled.constants, b"", fused.tobytes()],
                       offsets=[0, 0, 2 * _ZERO_TILE_WORDS], dump_bytes=fused.nbytes,
                       what=f"{heads} heads over {tokens} tokens", build_dir=build_dir, max_cycles=max_cycles)
    attended = _unpack(np.frombuffer(dumped, dtype="<u2"), tokens=tokens, features=3 * heads * _HEAD)

    mask = np.where(np.tril(np.ones((tokens, tokens), dtype=bool)), np.float32(0.0), np.float32(_MASKED))
    return _errors(attended, query, key, value, heads=heads, mask=mask, rows=tokens)


def simulate_decode(*, length: int, heads: int, build_dir: str | None = None,
                    max_cycles: int | None = None) -> tuple[float, float]:
    """Runs one decode step's every head against the RTL in one launch and returns what it and a BF16 model cost.

    Args:
        length: Keys the cache holds, which the image pads to whole token blocks and masks past.
        heads: Heads the fused tensor and the cache carry.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles the round may take, or None for the bound the testbench carries.

    Returns:
        The worst head's error on the new token against the exact reference, and the worst error the BF16 model
        carries.
    """
    keys = -(-length // _BLOCK_TOKENS) * _BLOCK_TOKENS
    compiled = compile_kernel(gpt2_decode_attention, heads=heads, N=keys, LENGTH=length)
    entry = compiled.entry
    generator = np.random.default_rng(1)

    # Stages the new token in the fused block's first row, the rest of the block being padding, and the cache's keys
    # and values past its length as values no query may attend to.
    fused = np.concatenate([_drawn(generator, rows=_BLOCK_TOKENS, heads=heads) for _ in range(3)], axis=1)
    key, value = _drawn(generator, rows=keys, heads=heads), _drawn(generator, rows=keys, heads=heads)
    masks = decode_mask_tiles(keys=keys, length=length)
    if entry["transposed_masks"]:
        masks = masks.swapaxes(-1, -2)
    constants = pack_attention_constants(initial_maximum=entry["entering"], constants=tuple(entry["stated"]),
                                         masks=masks).reshape(-1)
    packed, cache = _pack(fused), _pack(np.concatenate([key, value], axis=1))
    dumped = run_image(compiled.image, arguments=[constants.tobytes(), cache.tobytes(), packed.tobytes()],
                       offsets=[0, 2 * _ZERO_TILE_WORDS, 2 * _ZERO_TILE_WORDS], dump_bytes=packed.nbytes,
                       what=f"{heads} heads of one decode step over {length} keys", build_dir=build_dir,
                       max_cycles=max_cycles)
    attended = _unpack(np.frombuffer(dumped, dtype="<u2"), tokens=_BLOCK_TOKENS, features=3 * heads * _HEAD)

    mask = np.broadcast_to(np.where(np.arange(keys) < length, np.float32(0.0), np.float32(_MASKED)), (_TILE, keys))
    return _errors(attended, fused[:_TILE], key, value, heads=heads, mask=mask, rows=1)


def _drawn(generator: np.random.Generator, *, rows: int, heads: int) -> NDArray[np.float32]:
    """Returns operands on the BF16 grid, one row per token and one column per feature of every head."""
    return from_bf16_words(words=as_bf16_words(values=generator.uniform(
        -_SPREAD, _SPREAD, size=(rows, heads * _HEAD)).astype(np.float32)))


def _pack(matrix: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays a [tokens, features] tensor out as the device holds it: per token block, a zeroed tile and then its
    feature tiles, one row per token."""
    blocks = []
    for block in range(matrix.shape[0] // _BLOCK_TOKENS):
        rows = matrix[block * _BLOCK_TOKENS:(block + 1) * _BLOCK_TOKENS]
        blocks += [np.zeros(_ZERO_TILE_WORDS, dtype=np.float32),
                   rows.reshape(_BLOCK_TOKENS, -1, _TILE).transpose(1, 0, 2).reshape(-1)]
    return as_bf16_words(values=np.concatenate(blocks))


def _unpack(words: NDArray[np.uint16], *, tokens: int, features: int) -> NDArray[np.float32]:
    """Reads a tensor packed as `_pack` lays it out back into [tokens, features]."""
    block_words = _ZERO_TILE_WORDS + features * _BLOCK_TOKENS
    rows = []
    for block in range(tokens // _BLOCK_TOKENS):
        tiles = from_bf16_words(words=words[block * block_words + _ZERO_TILE_WORDS:(block + 1) * block_words])
        rows.append(tiles.reshape(features // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2)
                    .reshape(_BLOCK_TOKENS, -1))
    return np.concatenate(rows)


def _errors(attended: NDArray[np.float32], query: NDArray[np.float32], key: NDArray[np.float32],
            value: NDArray[np.float32], *, heads: int, mask: NDArray[np.float32], rows: int) -> tuple[float, float]:
    """Returns the worst head's error over its first `rows` queries against the exact reference, and the worst error
    the BF16 model carries over the same operands."""
    errors, modelled_errors = [], []
    for head in range(heads):
        columns = slice(head * _HEAD, (head + 1) * _HEAD)
        expected = attention_reference(query[:, columns], key[:, columns], value[:, columns], mask=mask)[:rows]
        modelled = attention_bf16_reference(query[:, columns], key[:, columns], value[:, columns],
                                            initial_maximum=_MASKED, mask=mask)[:rows]
        bound = float(np.abs(expected).max())
        errors.append(float(np.abs(attended[:rows, columns] - expected).max() / bound))
        modelled_errors.append(float(np.abs(modelled - expected).max() / bound))
    return max(errors), max(modelled_errors)


def main() -> int:
    """Runs every head of a sequence or of a decode step against the RTL and reports the worst head's error.

    Returns:
        Zero when every head held to what BF16 costs over the same operands, plus the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=64, help="tokens the sequence holds")
    parser.add_argument("--heads", type=int, default=12, help="heads the fused tensor carries")
    parser.add_argument("--decode", type=int, default=None, metavar="LENGTH",
                        help="run one decode step over a cache of this many keys instead of a whole sequence")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles the round may take, past the bound the testbench carries")
    arguments = parser.parse_args()

    if arguments.decode is None:
        error, modelled = simulate(tokens=arguments.tokens, heads=arguments.heads, build_dir=arguments.build_dir,
                                   max_cycles=arguments.max_cycles)
        what = f"{arguments.tokens} tokens"
    else:
        error, modelled = simulate_decode(length=arguments.decode, heads=arguments.heads,
                                          build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
        what = f"one decode step over {arguments.decode} keys"
    verdict = "PASS" if error <= modelled + _TOLERANCE else "FAIL"
    print(f"resident attention  {arguments.heads} heads, {what}, in one launch  {verdict}  "
          f"worst head error {error * 100:.4f}%, BF16 model {modelled * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
