#!/usr/bin/env python3
"""Provides the RTL check that an emitted GEMM image computes the product it claims.
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
from compiler.compile import Launch  # noqa: E402
from bf16 import (_BLOCK_DEPTH, _BLOCK_TOKENS, _BLOCK_WEIGHT_BYTES, _ELEMENT_BYTES, _TILE,  # noqa: E402
                  _ZERO_TILE_WORDS, build_operands, from_bf16_words, pack_activations, pack_weights)

_TOLERANCE: float = 0.02
"""Relative error the BF16 accumulate chain is allowed, which the board check also holds to."""

_DEFAULT_RTL: pathlib.Path = _HERE.parent.parent / "minitpu-tmp"
"""Checkout `MINITPU_RTL` falls back to, being a sibling of this repository."""


def rtl_root() -> pathlib.Path:
    """Returns the checkout carrying the RTL and its kernel image testbench.

    Returns:
        The checkout's root.

    Raises:
        FileNotFoundError: If the checkout does not carry `tb/run_kernel_image.sh`.
    """
    root = pathlib.Path(os.environ.get("MINITPU_RTL", _DEFAULT_RTL)).expanduser()
    if not (root / "tb" / "run_kernel_image.sh").is_file():
        message = (
            f"Unable to find the RTL under {root}. This repository does not vendor it, so set MINITPU_RTL to a "
            f"checkout carrying tb/run_kernel_image.sh."
        )
        raise FileNotFoundError(message)
    return root


def simulate(*, launch: Launch, reduction: int, build_dir: str | None = None) -> float:
    """Runs the GEMM one launch describes against the RTL and returns its error against a float64 reference.

    Args:
        launch: The launch to emit and run, covering one token block.
        reduction: The contraction the product reduces over, which the launch states in groups rather than elements.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one. Naming it keeps
            the build across runs, which is what makes a sweep cheap.

    Returns:
        The relative error against the host reference.

    Raises:
        RuntimeError: If the testbench does not report every round halted.
    """
    image = asm.AsmKernel(name="looped_gemm", bundles=emit.looped_gemm_stream(
        column_tiles=launch.column_tiles, groups=launch.groups, loop_blocks=launch.loop_blocks,
        fuse_groups=launch.fuse_groups, block_m=launch.block_m,
        block_n=launch.block_n)).to_bytes(pad_to=4)
    rows, columns = launch.block_m, launch.column_tiles * _TILE
    left, right = build_operands(rows, reduction, columns, kind="varying", seed=0)
    partials = 1 if launch.fuse_groups else launch.groups
    drain_words = _ZERO_TILE_WORDS + launch.column_tiles * partials * launch.block_m * _TILE

    with tempfile.TemporaryDirectory() as staging:
        stage = pathlib.Path(staging)
        (stage / "kernel.bin").write_bytes(image)
        (stage / "arg0.bin").write_bytes(pack_activations(activations=left).tobytes())
        slack = np.zeros(_BLOCK_WEIGHT_BYTES // _ELEMENT_BYTES, dtype="<u2")
        (stage / "arg1.bin").write_bytes(np.concatenate([pack_weights(weights=right), slack]).tobytes())
        (stage / "arg2.bin").write_bytes(np.zeros(drain_words, dtype="<u2").tobytes())

        environment = dict(os.environ)
        if build_dir is not None:
            environment["KERNEL_IMAGE_BUILD_DIR"] = str(pathlib.Path(build_dir).resolve())
        root = rtl_root()
        run = subprocess.run(
            ["bash", str(root / "tb" / "run_kernel_image.sh"),
             f"+KERNEL_BIN={stage / 'kernel.bin'}", f"+ARG0={stage / 'arg0.bin'}",
             f"+ARG1={stage / 'arg1.bin'}", f"+ARG2={stage / 'arg2.bin'}",
             f"+DUMP={stage / 'dump.bin'}", f"+DUMP_BYTES={drain_words * _ELEMENT_BYTES}"],
            cwd=root, env=environment, capture_output=True, text=True)
        if "=== TB_KERNEL_IMAGE PASS ===" not in run.stdout:
            rounds = "\n".join(line for line in run.stdout.splitlines() if "KERNEL_IMAGE_ROUND" in line)
            message = (
                f"Unable to complete the {rows}x{reduction} by {reduction}x{columns} product. The testbench did not "
                f"report every round halted.\n{rounds or run.stdout[-2000:]}\n{run.stderr[-2000:]}"
            )
            raise RuntimeError(message)
        dumped = np.frombuffer((stage / "dump.bin").read_bytes(), dtype="<u2")[:drain_words]

    # Reshapes by the blocks a launch covers, since a block drains its tiles together.
    tiles = launch.block_n // _TILE
    drained = from_bf16_words(words=dumped[_ZERO_TILE_WORDS:drain_words])
    shaped = drained.reshape(launch.column_tiles // tiles, partials, tiles, launch.block_m, _TILE)
    summed = shaped.sum(axis=1, dtype=np.float32).reshape(launch.column_tiles, launch.block_m, _TILE)
    product: NDArray[np.float32] = summed.transpose(1, 0, 2).reshape(launch.block_m,
                                                                    launch.column_tiles * _TILE)
    expected = (left.astype(np.float64) @ right.astype(np.float64)).astype(np.float32)
    return float(np.abs(product - expected).max() / np.abs(expected).max())


def main() -> int:
    """Runs one product against the RTL and reports its error.

    Returns:
        Zero when the product matched within the tolerance the BF16 accumulate chain allows.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=_BLOCK_TOKENS, help="token rows, being one block")
    parser.add_argument("--depth", type=int, default=192, help="the contraction, a multiple of the group depth")
    parser.add_argument("--columns", type=int, default=48, help="output columns, a multiple of the tile")
    parser.add_argument("--fuse-groups", action="store_true", help="accumulate the groups on the device")
    parser.add_argument("--loop-blocks", type=int, default=2, help="blocks one group carries")
    parser.add_argument("--block-m", type=int, default=_BLOCK_TOKENS, help="token rows one block covers")
    parser.add_argument("--block-n", type=int, default=_TILE, help="output columns one block covers")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    arguments = parser.parse_args()

    # States the launch, since these dimensions came from a person rather than from a plan.
    launch = Launch(row_blocks=arguments.rows // arguments.block_m, column_tiles=arguments.columns // _TILE,
                    groups=arguments.depth // (arguments.loop_blocks * _BLOCK_DEPTH),
                    loop_blocks=arguments.loop_blocks, activate=False, fuse_groups=arguments.fuse_groups,
                    block_m=arguments.block_m, block_n=arguments.block_n)
    error = simulate(launch=launch, reduction=arguments.depth, build_dir=arguments.build_dir)
    verdict = "PASS" if error <= _TOLERANCE else "FAIL"
    print(f"{arguments.rows}x{arguments.depth} @ {arguments.depth}x{arguments.columns} "
          f"fuse_groups={arguments.fuse_groups}  {verdict}  relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
