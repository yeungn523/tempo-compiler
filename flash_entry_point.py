#!/usr/bin/env python3
"""Compiles one attention kernel into the image and packed operands a board launches it from.

An attention block reads three operands, drains no clearing tile and walks its query blocks itself, so it is packed
here rather than through the entry point the products share.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The repository's root."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "board_package" / "minitpu_runtime"))

from asm import BUNDLE_BYTES, IRAM_DEPTH  # noqa: E402
from bf16 import (_ELEMENT_BYTES, attention_bf16_reference, attention_reference,  # noqa: E402
                  pack_attention)
from compiler import FlashLaunch, ProgramError, compile_launch, plan, read_program, signature_of  # noqa: E402
from kernel_entry_point import _bindings, _integer_bindings, read_kernel  # noqa: E402

def main() -> int:
    """Compiles one attention kernel and writes the image and operands a board launches it from.

    Returns:
        Zero once the image is written.

    Raises:
        ProgramError: If the kernel does not plan as one attention image, or if the operands are named in part.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True, metavar="FILE:NAME",
                        help="the Python file the kernel is declared in, naming the kernel after a colon")
    parser.add_argument("--operands", default="",
                        help="Triton type of each pointer parameter, for a kernel whose own annotation states none")
    parser.add_argument("--constexprs", default="", help="value of each tl.constexpr the kernel leaves open")
    parser.add_argument("--strides", default="", help="value of each stride parameter at the launch")
    parser.add_argument("--output", default=None, help="directory the image and operands are written to")
    parser.add_argument("--query", default=None, help="a .npy holding the queries one block covers")
    parser.add_argument("--key", default=None, help="a .npy holding every key the block attends over")
    parser.add_argument("--value", default=None, help="a .npy holding every value the block attends over")
    arguments = parser.parse_args()

    kernel = read_kernel(arguments.kernel)
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel,
                           signature=signature_of(kernel, operands=_bindings(arguments.operands,
                                                                            flag="--operands")),
                           constexprs=declared | _integer_bindings(arguments.constexprs, flag="--constexprs"))
    planned = plan(matmuls, strides=_integer_bindings(arguments.strides, flag="--strides"))

    launch = planned.launches[0]
    if len(planned.launches) != 1 or not isinstance(launch, FlashLaunch):
        stated = ", ".join(type(one).__name__ for one in planned.launches)
        message = (
            f"Unable to compile {arguments.kernel} as an attention image, which planned as {stated}. This entry "
            f"point covers a kernel whose two products an online softmax stands between, so compile any other "
            f"through kernel_entry_point.py."
        )
        raise ProgramError(message)

    image, staging = compile_launch(launch)
    drain = planned.drains[0]
    bundles = len(image) // BUNDLE_BYTES
    print(f"launch  row_blocks={launch.row_blocks} key_blocks={launch.key_blocks} "
          f"queries={launch.queries} head={launch.head} entering={launch.entering:g} "
          f"constants={len(launch.constants)} masked={launch.mask is not None}")
    print(f"image   {bundles} bundles over {staging.rows}x{staging.columns}, "
          f"{100 * bundles / IRAM_DEPTH:.0f}% of IRAM, draining {drain.words} words shaped {drain.shape}")

    named = [path for path in (arguments.query, arguments.key, arguments.value) if path is not None]
    if named and len(named) != 3:
        message = (
            "Unable to pack an attention block from some of its operands. The queries, the keys and the values "
            "are packed together, so name all three or none."
        )
        raise ProgramError(message)
    if arguments.output is None:
        return 0

    written = pathlib.Path(arguments.output).expanduser().resolve()
    written.mkdir(parents=True, exist_ok=True)
    (written / "attention.bin").write_bytes(image)
    print(f"wrote   {written / 'attention.bin'}  {len(image)} bytes")
    if named:
        _pack(path=written / "attention_operands.npz", launch=launch, drain=drain,
              query=np.load(arguments.query), key=np.load(arguments.key), value=np.load(arguments.value))
    return 0


def _pack(*, path: pathlib.Path, launch: FlashLaunch, drain, query: np.ndarray, key: np.ndarray,
          value: np.ndarray) -> None:
    """Writes the packed operands and every size a board allocates and reshapes from.

    Args:
        path: The file the operands are written to.
        launch: The launch the plan produced.
        drain: The drain the plan states.
        query: The queries one block covers.
        key: Every key the block attends over.
        value: Every value the block attends over.
    """
    staged, streamed = pack_attention(query, key, value, initial_maximum=launch.entering,
                                      constants=launch.constants, masks=launch.mask_tiles)
    expected = attention_reference(query, key, value, mask=launch.mask_scores).astype(np.float32)
    modelled = attention_bf16_reference(query, key, value, initial_maximum=launch.entering,
                                        mask=launch.mask_scores)
    np.savez(
        path,
        queries=staged,
        operands=streamed,
        expected=expected,
        # The bar a launch is held to. BF16 over two contractions and the online softmax between them costs
        # more than the tolerance a single product is allowed, and the board is too slow a core to derive that
        # cost there.
        format_cost=float(np.abs(modelled - expected).max() / max(float(np.abs(expected).max()), 1e-9)),
        constants=np.asarray(launch.constants, dtype=np.float32),
        row_blocks=launch.row_blocks,
        key_blocks=launch.key_blocks,
        head=launch.head,
        drain_words=drain.words,
        drain_bytes=drain.words * _ELEMENT_BYTES,
        zero_tile_words=drain.zero_tile_words,
        drain_shape=drain.shape,
    )
    print(f"wrote   {path}  queries {staged.size} words, keys and values {streamed.size} words")


if __name__ == "__main__":
    raise SystemExit(main())
