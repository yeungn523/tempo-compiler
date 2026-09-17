#!/usr/bin/env python3
"""Provides the RTL check that one attention block computes the attention a host computes.

The image is the one the compiler emits for a flash launch, so this checks the path a kernel reaches the board
through rather than a builder standing beside it."'
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

from bf16 import (_NUM_LANES, _TILE, _VREG_ELEMENTS, as_bf16_words,  # noqa: E402
                  attention_bf16_reference, from_bf16_words)
from compiler import FlashLaunch, compile_launch  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the block is allowed past what the BF16 format costs over the same two contractions."""

_INITIAL_MAXIMUM: float = -1e4
"""Value the running maximum enters below, which every score clears and whose exponential is zero."""

_SPREAD: float = 2.0
"""Bound the drawn operands fall inside, which keeps the scores inside the range the table covers."""

_SEED: int = 0
"""Seed the operands are drawn under, so a failure is reproducible."""


def _tiled(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns one operand as the 16 wide tiles the image stages, each tile a run of 16 rows.

    Args:
        matrix: The operand, whose columns are a whole number of tiles.

    Returns:
        The stacked tiles.
    """
    width = matrix.shape[1]
    return np.concatenate([matrix[:, tile * _TILE: (tile + 1) * _TILE]
                           for tile in range(width // _TILE)], axis=0)


def _operands(*, key_blocks: int, head: int, seed: int) -> tuple[NDArray[np.float32], ...]:
    """Returns the queries, keys and values the check runs on, at the precision the device holds them.

    Args:
        key_blocks: Key blocks the loop walks.
        head: Width of the head.
        seed: Seed the operands are drawn under.

    Returns:
        The queries, the keys and the values.
    """
    generator = np.random.default_rng(seed)
    keys = key_blocks * _TILE
    drawn = [generator.uniform(-_SPREAD, _SPREAD, size=shape).astype(np.float32)
             for shape in ((_TILE, head), (keys, head), (keys, head))]
    return tuple(from_bf16_words(as_bf16_words(operand)).reshape(operand.shape) for operand in drawn)


def simulate(*, key_blocks: int = 2, head: int = _VREG_ELEMENTS, seed: int = _SEED,
             build_dir: str | None = None, max_cycles: int | None = None) -> tuple[float, float]:
    """Runs one attention block against the RTL and returns what it and a BF16 model each cost.

    Args:
        key_blocks: Key blocks the loop walks, each covering 16 keys.
        head: Width of the head the block accumulates.
        seed: Seed the operands are drawn under.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles the round may take, or None for the bound the testbench carries.

    Returns:
        The block's error against the exact reference, and the error a model rounding where the machine rounds
        carries over the same operands.

    Raises:
        RuntimeError: If the testbench does not report the round halted.
    """
    image, _ = compile_launch(FlashLaunch(row_blocks=1, key_blocks=key_blocks, queries=_TILE, head=head))
    query, key, value = _operands(key_blocks=key_blocks, head=head, seed=seed)

    initial = np.full((_TILE, _NUM_LANES), _INITIAL_MAXIMUM, dtype=np.float32)
    staged = np.concatenate([_tiled(query), initial, np.zeros((_TILE, _NUM_LANES), dtype=np.float32)], axis=0)

    blocks = []
    for block in range(key_blocks):
        rows = slice(block * _TILE, (block + 1) * _TILE)
        transposed = np.concatenate([key[rows, tile * _TILE: (tile + 1) * _TILE].T
                                     for tile in range(head // _TILE)], axis=0)
        blocks.extend([transposed, _tiled(value[rows])])

    dump = _run_image(image=image,
                      activations=as_bf16_words(staged).tobytes(),
                      weights=as_bf16_words(np.concatenate(blocks, axis=0)).tobytes(),
                      drain_bytes=head * _NUM_LANES * 2, build_dir=build_dir, max_cycles=max_cycles,
                      what=f"attention over {key_blocks} key blocks of a head of {head}")
    drained = from_bf16_words(np.frombuffer(dump, dtype="<u2")[: head * _NUM_LANES])
    attended = np.concatenate(drained.reshape(head // _TILE, _TILE, _NUM_LANES), axis=1)

    scores = query.astype(np.float64) @ key.astype(np.float64).T
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    expected = (weights / weights.sum(axis=1, keepdims=True)) @ value.astype(np.float64)
    modelled = attention_bf16_reference(query, key, value, initial_maximum=_INITIAL_MAXIMUM)
    scale = np.abs(expected).max()
    return (float(np.abs(attended - expected).max() / scale),
            float(np.abs(modelled - expected).max() / scale))


def main() -> int:
    """Runs one attention block against the RTL and reports its error.

    Returns:
        Zero when the block held to what BF16 costs over the same operands, plus the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-blocks", type=int, default=2, help="key blocks the loop walks")
    parser.add_argument("--head", type=int, default=_VREG_ELEMENTS, help="width of the head")
    parser.add_argument("--seed", type=int, default=_SEED, help="seed the operands are drawn under")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles the round may take, past the bound the testbench carries")
    arguments = parser.parse_args()

    error, modelled = simulate(key_blocks=arguments.key_blocks, head=arguments.head, seed=arguments.seed,
                               build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
    verdict = "PASS" if error <= modelled + _TOLERANCE else "FAIL"
    print(f"attention  {arguments.key_blocks} key blocks, head {arguments.head}  {verdict}  "
          f"error {error * 100:.4f}%, BF16 model {modelled * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
