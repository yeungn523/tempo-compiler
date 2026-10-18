#!/usr/bin/env python3
"""Provides the RTL check that a GEMM image computes the product it claims.

The image comes either from the dimensions named on the command line, which emits one directly, or from the
artifacts `kernel_entry_point.py` wrote for a Triton kernel, which is the path a compiled kernel takes. Both end in the
same testbench, so a kernel is checked here before it reaches hardware.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The repository's tools directory."""

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "board_package"))
sys.path.insert(0, str(_HERE.parent / "board_package" / "minitpu_runtime"))

import asm  # noqa: E402
import emit  # noqa: E402
from bf16 import (_BLOCK_DEPTH, _BLOCK_TOKENS, _BLOCK_WEIGHT_BYTES, _ELEMENT_BYTES, _TILE,  # noqa: E402
                  _ZERO_TILE_WORDS, build_operands, from_bf16_words, pack_activations, pack_weights,
                  reduce_drain)
from compiler.ir import Launch  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the BF16 accumulate chain is allowed, which the board check also holds to."""

_DEFAULT_RTL: pathlib.Path = _HERE.parent.parent / "minitpu-tmp"
"""Checkout `minitpu_rtl` falls back to, being a sibling of this repository."""

_PASS_MARKER: str = "=== TB_KERNEL_IMAGE PASS ==="
"""Line the testbench prints once every round has halted."""


def simulate(*, launch: Launch, reduction: int, build_dir: str | None = None,
             max_cycles: int | None = None) -> float:
    """Runs the GEMM one launch describes against the RTL and returns its error against a float64 reference.

    Args:
        launch: The launch to emit and run, covering one token block.
        reduction: The contraction the product reduces over, which the launch states in groups rather than elements.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one. Naming it keeps
            the build across runs, which is what makes a sweep cheap.
        max_cycles: Cycles one round may take before the testbench reports it timed out, or None for the bound the
            testbench carries.

    Returns:
        The relative error against the host reference.

    Raises:
        RuntimeError: If the testbench does not report every round halted.
    """
    counts = dict(column_tiles=launch.column_tiles, groups=launch.groups, loop_blocks=launch.loop_blocks,
                  block_m=launch.block_m, block_n=launch.block_n)
    # Takes every shape from the emitter, which settled them placing the bundles, so this states none of its own.
    bundles, staging = emit.looped_gemm_stream(**counts)
    image = asm.AsmKernel(name="looped_gemm", bundles=bundles).to_bytes(pad_to=4)
    left, right = build_operands(staging.rows, reduction, staging.columns, kind="varying", seed=0)
    drain_words = staging.drain_words

    slack = np.zeros(_BLOCK_WEIGHT_BYTES // _ELEMENT_BYTES, dtype="<u2")
    dump = _run_image(image=image, activations=pack_activations(activations=left).tobytes(),
                      weights=np.concatenate([pack_weights(weights=right), slack]).tobytes(),
                      drain_bytes=drain_words * _ELEMENT_BYTES, build_dir=build_dir, max_cycles=max_cycles,
                      what=f"the {staging.rows}x{reduction} by {reduction}x{staging.columns} product")
    dumped = np.frombuffer(dump, dtype="<u2")[:drain_words]

    drained = from_bf16_words(words=dumped[_ZERO_TILE_WORDS:drain_words])
    product: NDArray[np.float32] = reduce_drain(drained, shape=(
        staging.blocks, staging.partials, staging.output_tiles, staging.rows, _TILE))
    expected = (left.astype(np.float64) @ right.astype(np.float64)).astype(np.float32)
    return float(np.abs(product - expected).max() / np.abs(expected).max())


def simulate_compiled(*, image: bytes, operands: np.lib.npyio.NpzFile, build_dir: str | None = None,
                      max_cycles: int | None = None) -> float:
    """Runs a compiled image against the RTL once per token block and returns its error.

    Args:
        image: The kernel image the entry point emitted.
        operands: The packed operands and sizes the entry point wrote beside it.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take before the testbench reports it timed out, or None for the bound the
            testbench carries.

    Returns:
        The relative error against the reference the host computed.

    Raises:
        RuntimeError: If the testbench does not report every round halted.
    """
    activations = operands["activations"]
    weights = operands["weights"]
    drain_words = int(operands["drain_words"])
    zero_tile_words = int(operands["zero_tile_words"])
    shape = tuple(int(extent) for extent in operands["drain_shape"])
    block_columns = int(operands["block_columns"])
    block_rows = shape[3]

    expected: NDArray[np.float32] = operands["expected"]
    product: NDArray[np.float32] = np.empty(expected.shape, dtype=np.float32)
    for token, column in _issued(order=str(operands["issue_order"]), tokens=len(activations),
                                 columns=len(weights)):
        packed = weights[column].tobytes()
        dump = _run_image(image=image, activations=activations[token].tobytes(),
                          weights=packed + bytes(int(operands["weight_bytes"]) - len(packed)),
                          drain_bytes=int(operands["drain_bytes"]), build_dir=build_dir, max_cycles=max_cycles,
                          what=f"token block {token} column block {column}")
        dumped = np.frombuffer(dump, dtype="<u2")[:drain_words]
        piece = reduce_drain(from_bf16_words(words=dumped[zero_tile_words:drain_words]), shape=shape)
        product[token * block_rows : (token + 1) * block_rows,
                column * block_columns : (column + 1) * block_columns] = piece
    return float(np.abs(product - expected).max() / np.abs(expected).max())


def _issued(*, order: str, tokens: int, columns: int) -> list[tuple[int, int]]:
    """Returns every launch a host issues, as its token block and column block, in the order the kernel walks them.

    Args:
        order: The output axes the kernel walks, outermost first, as the entry point packed them.
        tokens: Token blocks the product covers.
        columns: Column blocks one token block is issued against.

    Returns:
        The token block and column block of each launch, in issue order.
    """
    if order.startswith("n"):
        return [(token, column) for column in range(columns) for token in range(tokens)]
    return [(token, column) for token in range(tokens) for column in range(columns)]


def _run_image(*, image: bytes, activations: bytes, weights: bytes, drain_bytes: int,
               build_dir: str | None, max_cycles: int | None, what: str, drain: bytes | None = None) -> bytes:
    """Runs one image against the RTL with the operands a launch reads, returning the bytes it drained.

    Args:
        image: The kernel image, padded to the loader's boundary.
        activations: The packed activation block the launch reads as its first argument.
        weights: The packed weights the launch reads as its second argument, slack included.
        drain_bytes: Bytes the drain buffer holds, which is what the testbench dumps.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take before the testbench reports it timed out, or None for the bound the
            testbench carries.
        what: The product being run, for the message a failed round carries.
        drain: What the drain buffer holds before the launch, or None for zeros.

    Returns:
        The drained bytes.

    Raises:
        RuntimeError: If the testbench does not report every round halted.
    """
    with tempfile.TemporaryDirectory() as scratch:
        stage = pathlib.Path(scratch)
        (stage / "kernel.bin").write_bytes(image)
        (stage / "arg0.bin").write_bytes(activations)
        (stage / "arg1.bin").write_bytes(weights)
        (stage / "arg2.bin").write_bytes(bytes(drain_bytes) if drain is None else drain)

        environment = dict(os.environ)
        if build_dir is not None:
            environment["KERNEL_IMAGE_BUILD_DIR"] = str(pathlib.Path(build_dir).resolve())
        root = _rtl_root()
        bound = [f"+MAX_CYCLES={max_cycles}"] if max_cycles is not None else []
        run = subprocess.run(
            ["bash", str(root / "tb" / "run_kernel_image.sh"),
             f"+KERNEL_BIN={stage / 'kernel.bin'}", f"+ARG0={stage / 'arg0.bin'}",
             f"+ARG1={stage / 'arg1.bin'}", f"+ARG2={stage / 'arg2.bin'}",
             f"+DUMP={stage / 'dump.bin'}", f"+DUMP_BYTES={drain_bytes}", *bound],
            cwd=root, env=environment, capture_output=True, text=True)
        if _PASS_MARKER not in run.stdout:
            rounds = "\n".join(line for line in run.stdout.splitlines() if "KERNEL_IMAGE_ROUND" in line)
            message = (
                f"Unable to complete {what}. The testbench did not report every round halted.\n"
                f"{rounds or run.stdout[-2000:]}\n{run.stderr[-2000:]}"
            )
            raise RuntimeError(message)
        return (stage / "dump.bin").read_bytes()


def _rtl_root() -> pathlib.Path:
    """Returns the checkout carrying the RTL and its kernel image testbench.

    Returns:
        The checkout's root.

    Raises:
        FileNotFoundError: If the checkout does not carry `tb/run_kernel_image.sh`.
    """
    root = pathlib.Path(os.environ.get("minitpu_rtl", _DEFAULT_RTL)).expanduser()
    if not (root / "tb" / "run_kernel_image.sh").is_file():
        message = (
            f"Unable to find the RTL under {root}. This repository does not vendor it, so set minitpu_rtl to a "
            f"checkout carrying tb/run_kernel_image.sh."
        )
        raise FileNotFoundError(message)
    return root


def main() -> int:
    """Runs one product against the RTL and reports its error.

    Returns:
        Zero when the product matched within the tolerance the BF16 accumulate chain allows.

    Raises:
        ValueError: If the image and the operands are not named together.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=None, help="a .bin the entry point emitted, with --operands")
    parser.add_argument("--operands", default=None, help="the _operands.npz packed beside that image")
    parser.add_argument("--rows", type=int, default=_BLOCK_TOKENS, help="token rows, being one block")
    parser.add_argument("--depth", type=int, default=192, help="the contraction, a multiple of the group depth")
    parser.add_argument("--columns", type=int, default=48, help="output columns, a multiple of the tile")
    parser.add_argument("--loop-blocks", type=int, default=2, help="blocks one group carries")
    parser.add_argument("--block-m", type=int, default=_BLOCK_TOKENS, help="token rows one block covers")
    parser.add_argument("--block-n", type=int, default=_TILE, help="output columns one block covers")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles one round may take, past the bound the testbench carries")
    arguments = parser.parse_args()

    if (arguments.image is None) != (arguments.operands is None):
        message = (
            "Unable to run one artifact alone. The image names the bundles and the operands name the sizes they "
            "run against, so name both or neither."
        )
        raise ValueError(message)

    if arguments.image is not None:
        operands = np.load(arguments.operands)
        error = simulate_compiled(image=pathlib.Path(arguments.image).read_bytes(), operands=operands,
                                  build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
        rows, columns = operands["expected"].shape
        what = f"{rows}x{columns} product over {len(operands['activations'])} token blocks"
    else:
        # States the launch, since these dimensions came from a person rather than from a plan.
        launch = Launch(row_blocks=arguments.rows // arguments.block_m, column_tiles=arguments.columns // _TILE,
                        groups=arguments.depth // (arguments.loop_blocks * _BLOCK_DEPTH),
                        loop_blocks=arguments.loop_blocks, activate=False,
                        block_m=arguments.block_m, block_n=arguments.block_n)
        error = simulate(launch=launch, reduction=arguments.depth, build_dir=arguments.build_dir,
                         max_cycles=arguments.max_cycles)
        what = f"{arguments.rows}x{arguments.depth} @ {arguments.depth}x{arguments.columns}"

    verdict = "PASS" if error <= _TOLERANCE else "FAIL"
    print(f"{what}  {verdict}  relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
