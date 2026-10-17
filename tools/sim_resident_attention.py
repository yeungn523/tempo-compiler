#!/usr/bin/env python3
"""Provides the RTL check that the resident attention images attend every head of a sequence, or of a decode step,
in one launch.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The repository's tools directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "board_package"))
sys.path.insert(0, str(_HERE.parent / "board_package" / "minitpu_runtime"))

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _ZERO_TILE_WORDS,
    as_bf16_words,
    attention_bf16_reference,
    attention_reference,
    from_bf16_words,
    pack_activations,
    pack_attention_constants,
)
from compiler import FlashLaunch, compile_resident_attention, plan, read_program, signature_of  # noqa: E402
from gpt2_triton import gpt2_attention, gpt2_decode_attention  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the head is allowed past what the BF16 model costs over the same operands."""

_HEAD: int = 64
"""Width of one attention head, being GPT-2 small's."""

_SPREAD: float = 1.25
"""Bound the operands are drawn inside, which keeps the scores inside the range the exponential table covers."""


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
    launch = _planned(tokens=tokens)
    fused_tiles = 3 * heads * _HEAD // _TILE
    image = compile_resident_attention(launch, heads=heads, fused_tiles=fused_tiles)
    generator = np.random.default_rng(0)
    query, key, value = (from_bf16_words(words=as_bf16_words(values=generator.uniform(
        -_SPREAD, _SPREAD, size=(tokens, heads * _HEAD)).astype(np.float32))) for _ in range(3))

    # Lays the fused tensor out as resident blocks do, each block's packed tokens followed by the next block's
    # clearing tile, from the first block's tokens onward, which is where the launch's argument points.
    fused = np.concatenate([query, key, value], axis=1)
    blocks = [pack_activations(activations=fused[block * _BLOCK_TOKENS: (block + 1) * _BLOCK_TOKENS])
              for block in range(tokens // _BLOCK_TOKENS)]
    clearing = np.zeros(_ZERO_TILE_WORDS, dtype=np.uint16)
    resident = np.concatenate([piece for block in blocks for piece in (block, clearing)])
    constants = pack_attention_constants(initial_maximum=launch.entering, constants=launch.constants,
                                         masks=launch.mask_tiles)
    dumped = _run_image(image=image, activations=constants.tobytes(), weights=b"", drain=resident.tobytes(),
                        drain_bytes=resident.nbytes, build_dir=build_dir, max_cycles=max_cycles,
                        what=f"{heads} heads over {tokens} tokens")
    words = np.frombuffer(dumped, dtype="<u2")
    block_words = blocks[0].size + _ZERO_TILE_WORDS
    attended = np.concatenate([
        from_bf16_words(words=words[block * block_words: block * block_words + heads * _HEAD * _BLOCK_TOKENS])
        .reshape(heads * _HEAD // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, -1)
        for block in range(len(blocks))], axis=0)

    mask = launch.mask_scores
    errors, modelled_errors = [], []
    for head in range(heads):
        columns = slice(head * _HEAD, (head + 1) * _HEAD)
        expected = attention_reference(query[:, columns], key[:, columns], value[:, columns], mask=mask)
        modelled = attention_bf16_reference(query[:, columns], key[:, columns], value[:, columns],
                                            initial_maximum=launch.entering, mask=mask)
        bound = float(np.abs(expected).max())
        errors.append(float(np.abs(attended[:, columns] - expected).max() / bound))
        modelled_errors.append(float(np.abs(modelled - expected).max() / bound))
    return max(errors), max(modelled_errors)


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
    padded = -(-length // _BLOCK_TOKENS) * _BLOCK_TOKENS
    declared = {parameter.name: parameter.default for parameter in gpt2_decode_attention.params
                if parameter.is_constexpr}
    matmuls = read_program(gpt2_decode_attention, signature=signature_of(gpt2_decode_attention, operands={}),
                           constexprs=declared | {"N": padded, "LENGTH": length})
    launch = plan(matmuls, strides={}).launches[0]
    model_tiles = heads * _HEAD // _TILE
    image = compile_resident_attention(launch, heads=heads, fused_tiles=3 * model_tiles, cache_tiles=2 * model_tiles)
    generator = np.random.default_rng(1)

    def drawn(rows: int) -> NDArray[np.float32]:
        """Returns operands on the BF16 grid, one row per token and one column per feature of every head."""
        return from_bf16_words(words=as_bf16_words(values=generator.uniform(
            -_SPREAD, _SPREAD, size=(rows, heads * _HEAD)).astype(np.float32)))

    # Stages the new token in the fused block's first row, the rest of the block being padding, and the cache's
    # keys and values past its length as values no query may attend to.
    fused = np.concatenate([drawn(_BLOCK_TOKENS) for _ in range(3)], axis=1)
    key, value = drawn(padded), drawn(padded)
    cache = np.concatenate([key, value], axis=1)
    clearing = np.zeros(_ZERO_TILE_WORDS, dtype=np.uint16)
    resident = np.concatenate([piece for block in range(padded // _BLOCK_TOKENS) for piece in (
        pack_activations(activations=cache[block * _BLOCK_TOKENS: (block + 1) * _BLOCK_TOKENS]), clearing)])
    constants = pack_attention_constants(initial_maximum=launch.entering, constants=launch.constants,
                                         masks=launch.mask_tiles)
    packed = pack_activations(activations=fused)
    dumped = _run_image(image=image, activations=constants.tobytes(), weights=resident.tobytes(),
                        drain=packed.tobytes(), drain_bytes=packed.nbytes, build_dir=build_dir,
                        max_cycles=max_cycles, what=f"{heads} heads of one decode step over {length} keys")
    words = from_bf16_words(words=np.frombuffer(dumped, dtype="<u2")[:model_tiles * _BLOCK_TOKENS * _TILE])
    attended = words.reshape(model_tiles, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, -1)

    mask = launch.mask_scores
    errors, modelled_errors = [], []
    for head in range(heads):
        columns = slice(head * _HEAD, (head + 1) * _HEAD)
        queries = fused[:_TILE, columns]
        expected = attention_reference(queries, key[:, columns], value[:, columns], mask=mask)[:1]
        modelled = attention_bf16_reference(queries, key[:, columns], value[:, columns],
                                            initial_maximum=launch.entering, mask=mask)[:1]
        bound = float(np.abs(expected).max())
        errors.append(float(np.abs(attended[:1, columns] - expected).max() / bound))
        modelled_errors.append(float(np.abs(modelled - expected).max() / bound))
    return max(errors), max(modelled_errors)


def _planned(*, tokens: int) -> FlashLaunch:
    """Returns the launch GPT-2's attention kernel plans to over a sequence of the given length."""
    declared = {parameter.name: parameter.default for parameter in gpt2_attention.params if parameter.is_constexpr}
    matmuls = read_program(gpt2_attention, signature=signature_of(gpt2_attention, operands={}),
                           constexprs=declared | {"M": tokens})
    return plan(matmuls, strides={}).launches[0]


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
