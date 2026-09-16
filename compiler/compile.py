"""Provides the image one launch compiles to and the operands a host stages for it."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asm import AsmKernel
from bf16 import _BLOCK_DEPTH, _TILE
from emit import Staging, fused_ffn_stream, looped_gemm_stream

from .ir import FusedLaunch, Launch, MatMul, ProgramError


def stage(left: NDArray[np.float32], right: NDArray[np.float32], *, matmul: MatMul,
          launch: Launch) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Pads an operand pair to the extent its launch covers.

    Args:
        left: The activation operand at the extent the caller holds.
        right: The weight operand at the extent the caller holds.
        matmul: The matmul the kernel computes, whose tile gives the covered extents.
        launch: The planned launch, whose block and trip counts give the covered extents.

    Returns:
        The staged pair.
    """
    rows = launch.row_blocks * launch.block_m
    depth = launch.groups * launch.loop_blocks * _BLOCK_DEPTH
    columns = launch.column_tiles * _TILE
    staged = []
    for operand, shape in ((left, (rows, depth)), (right, (depth, columns))):
        # Slices before measuring, so an operand larger than its launch covers is cut rather than broadcast.
        kept = operand[: shape[0], : shape[1]]
        padded = np.zeros(shape, dtype=operand.dtype)
        padded[: kept.shape[0], : kept.shape[1]] = kept
        staged.append(padded)
    return staged[0], staged[1]


def compile_launch(launch: Launch | FusedLaunch) -> tuple[bytes, Staging]:
    """Returns the image one launch compiles to and what a host stages for it.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary, and the staging the emitter placed them from.

    Raises:
        NotImplementedError: If a plain launch activates its staged tiles, which the emitter does not place.
        ProgramError: If the compiled bundles exceed the 4096 IRAM depth limit.
    """
    if isinstance(launch, FusedLaunch):
        bundles, staging = fused_ffn_stream(chunks=launch.chunks, steps=launch.steps,
                                            output_tiles=launch.output_tiles)
        try:
            return AsmKernel(name="fused_ffn", bundles=bundles).to_bytes(pad_to=4), staging
        except ValueError as error:
            try:
                return AsmKernel(name="fused_ffn", bundles=bundles, placement="pack").to_bytes(pad_to=4), staging
            except ValueError:
                pass
            message = (
                f"Fused launch cannot fit in hardware IRAM: {error}\n"
                f"Note: Kernel image ({len(bundles)} bundles) exceeds the 4096 IRAM capacity."
            )
            raise ProgramError(message) from error

    if launch.activate:
        message = (
            "Unable to compile a launch that activates its staged tiles. The emitter places the staging, the "
            "matrix commands and the drain only, so such a launch comes from `kernels.build_looped_gemm` until "
            "the activation is placed here too."
        )
        raise NotImplementedError(message)
    counts = dict(column_tiles=launch.image_blocks * launch.output_tiles, groups=launch.groups,
                  loop_blocks=launch.loop_blocks,
                  fuse_groups=launch.fuse_groups, block_m=launch.block_m, block_n=launch.block_n)
    bundles, staging = looped_gemm_stream(**counts)
    try:
        return AsmKernel(name="looped_gemm", bundles=bundles).to_bytes(pad_to=4), staging
    except ValueError as error:
        try:
            return AsmKernel(name="looped_gemm", bundles=bundles, placement="pack").to_bytes(pad_to=4), staging
        except ValueError:
            pass
        message = (
            f"Launch cannot fit in hardware IRAM: {error}\n"
            f"Note: Unrolling {launch.output_tiles} output tiles for BLOCK_N={launch.block_n} "
            f"causes excessive code bloat ({len(bundles)} bundles > 4096 IRAM capacity). "
            f"Try a narrower tile width (such as 16, 32, or 64) to fit within instruction memory."
        )
        raise ProgramError(message) from error
