#!/usr/bin/env python3
"""Provides the RTL check that one attention block computes the attention a host computes.

The image is the one the compiler emits for a flash launch, so this checks the path a kernel reaches the board
through rather than a builder standing beside it."'
"""

from __future__ import annotations

import argparse
import dataclasses
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
                  attention_bf16_reference, from_bf16_words, pack_attention, staged_maximum)
from compiler import FlashLaunch, compile_launch, read_program, signature_of  # noqa: E402
from compiler.allocation import allocate  # noqa: E402
from compiler.dataflow import attention_trip  # noqa: E402
from compiler.lowering import attention_body, attention_epilogue  # noqa: E402
from kernel_entry_point import read_kernel  # noqa: E402
from layout import flash_register_map  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the block is allowed past what the BF16 format costs over the same two contractions."""

_SPREAD: float = 2.0
"""Bound the drawn operands fall inside, which keeps the scores inside the range the table covers."""

_SEED: int = 0
"""Seed the operands are drawn under, so a failure is reproducible."""

_KERNEL: str = str(_HERE.parent / "sample_flashattention.py") + ":flash_attention_kernel"
"""The kernel the steps are derived from, this check running the image a written kernel compiles to."""


def _operands(*, key_blocks: int, row_blocks: int, head: int,
              seed: int) -> tuple[NDArray[np.float32], ...]:
    """Returns the queries, keys and values the check runs on, at the precision the device holds them.

    Args:
        key_blocks: Key blocks the loop walks.
        row_blocks: Query blocks the image walks.
        head: Width of the head.
        seed: Seed the operands are drawn under.

    Returns:
        The queries, the keys and the values.
    """
    generator = np.random.default_rng(seed)
    keys = key_blocks * _TILE
    drawn = [generator.uniform(-_SPREAD, _SPREAD, size=shape).astype(np.float32)
             for shape in ((row_blocks * _TILE, head), (keys, head), (keys, head))]
    return tuple(from_bf16_words(as_bf16_words(operand)).reshape(operand.shape) for operand in drawn)


def simulate(*, key_blocks: int = 2, row_blocks: int = 1, head: int = _VREG_ELEMENTS, seed: int = _SEED,
             build_dir: str | None = None, max_cycles: int | None = None, kernel_name: str = _KERNEL,
             scale: float = 1.0) -> tuple[float, float]:
    """Runs one attention block against the RTL and returns what it and a BF16 model each cost.

    Args:
        key_blocks: Key blocks the loop walks, each covering 16 keys.
        row_blocks: Query blocks the image walks itself, each covering 16 queries.
        head: Width of the head the block accumulates.
        seed: Seed the operands are drawn under.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles the round may take, or None for the bound the testbench carries.
        kernel_name: The Python file the kernel is declared in, naming the kernel after a colon.
        scale: The scale the kernel applies to its scores, which the reference folds into the queries.

    Returns:
        The block's error against the exact reference, and the error a model rounding where the machine rounds
        carries over the same operands.

    Raises:
        RuntimeError: If the testbench does not report the round halted.
    """
    placed = _placed(kernel_name=kernel_name, head=head)
    image, _ = compile_launch(dataclasses.replace(placed, key_blocks=key_blocks, row_blocks=row_blocks))
    query, key, value = _operands(key_blocks=key_blocks, row_blocks=row_blocks, head=head, seed=seed)

    staged, streamed = pack_attention(query, key, value, initial_maximum=placed.entering,
                                      constants=placed.constants)

    words = row_blocks * head * _NUM_LANES
    dump = _run_image(image=image,
                      activations=staged.tobytes(),
                      weights=streamed.tobytes(),
                      drain_bytes=words * 2, build_dir=build_dir, max_cycles=max_cycles,
                      what=f"attention over {key_blocks} key blocks of a head of {head}")
    drained = from_bf16_words(np.frombuffer(dump, dtype="<u2")[:words])
    # One block drains its output a tile at a time, and the blocks follow one another.
    attended = np.concatenate([np.concatenate(block, axis=1) for block in
                               drained.reshape(row_blocks, head // _TILE, _TILE, _NUM_LANES)], axis=0)

    scores = scale * (query.astype(np.float64) @ key.astype(np.float64).T)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    expected = (weights / weights.sum(axis=1, keepdims=True)) @ value.astype(np.float64)
    modelled = attention_bf16_reference((scale * query).astype(np.float32), key, value,
                                        initial_maximum=placed.entering)
    bound = np.abs(expected).max()
    return (float(np.abs(attended - expected).max() / bound),
            float(np.abs(modelled - expected).max() / bound))


def _placed(*, kernel_name: str, head: int) -> FlashLaunch:
    """Returns the launch one kernel's trip places into at one head width.

    Args:
        kernel_name: The Python file the kernel is declared in, naming the kernel after a colon.
        head: Width of the head the block accumulates.

    Returns:
        A launch carrying the steps, the stated numbers and the value the maximum enters at.
    """
    kernel = read_kernel(kernel_name)
    constexprs = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands={}), constexprs=constexprs)
    trip = attention_trip(matmuls, produces=0, consumes=1)
    registers = flash_register_map(queries=_TILE, head=head)
    placement = allocate(trip, registers=registers)
    entering = trip.values[trip.carried["row_max"][0]].number
    return FlashLaunch(
        row_blocks=1, key_blocks=1, queries=_TILE, head=head,
        body=attention_body(placement, trip=trip, steps=head // _TILE, tiles=registers.output_tiles),
        epilogue=attention_epilogue(trip, tiles=registers.output_tiles), constants=trip.constants,
        entering=staged_maximum(0.0 if entering is None else entering))


def main() -> int:
    """Runs one attention block against the RTL and reports its error.

    Returns:
        Zero when the block held to what BF16 costs over the same operands, plus the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-blocks", type=int, default=2, help="key blocks the loop walks")
    parser.add_argument("--row-blocks", type=int, default=1, help="query blocks the image walks itself")
    parser.add_argument("--head", type=int, default=_VREG_ELEMENTS, help="width of the head")
    parser.add_argument("--seed", type=int, default=_SEED, help="seed the operands are drawn under")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles the round may take, past the bound the testbench carries")
    parser.add_argument("--kernel", default=_KERNEL, metavar="FILE:NAME",
                        help="the Python file the kernel is declared in, naming the kernel after a colon")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="the scale the kernel applies to its scores, which the reference folds in")
    arguments = parser.parse_args()

    error, modelled = simulate(key_blocks=arguments.key_blocks, row_blocks=arguments.row_blocks,
                               head=arguments.head, seed=arguments.seed,
                               build_dir=arguments.build_dir, max_cycles=arguments.max_cycles,
                               kernel_name=arguments.kernel, scale=arguments.scale)
    verdict = "PASS" if error <= modelled + _TOLERANCE else "FAIL"
    print(f"attention  {arguments.key_blocks} key blocks, {arguments.row_blocks} query blocks, "
          f"head {arguments.head}  {verdict}  "
          f"error {error * 100:.4f}%, BF16 model {modelled * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
