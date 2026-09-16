"""Provides the image one launch compiles to and the operands a host stages for it."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asm import IRAM_DEPTH, AsmKernel
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


def _assemble(*, name: str, bundles: tuple[int, ...], note: str) -> bytes:
    """Returns one kernel's image, packing its bundles where the hand placement overflows IRAM.

    The hand placement pads with empty bundles that carry no wait, so a stream it cannot fit is worth packing
    before it is refused. Both placements are measured rather than predicted, the image an emitted stream places
    being what decides whether a tile fits.

    Args:
        name: The name the kernel's error messages carry.
        bundles: The bundle words the emitter placed, in program order.
        note: What a refusal says the launch could not do, standing ahead of the count that proves it.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If neither placement fits the image in IRAM.
    """
    overflow: ValueError | None = None
    for placement in (None, "pack"):
        try:
            return AsmKernel(name=name, bundles=bundles, placement=placement).to_bytes(pad_to=4)
        except ValueError as error:
            overflow = error
    message = (
        f"{note}: {overflow}\n"
        f"Note: its {len(bundles)} bundles exceed the {IRAM_DEPTH} IRAM holds, packed as well as placed by hand."
    )
    raise ProgramError(message) from overflow


def compile_launch(launch: Launch | FusedLaunch) -> tuple[bytes, Staging]:
    """Returns the image one launch compiles to and what a host stages for it.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary, and the staging the emitter placed them from.

    Raises:
        NotImplementedError: If a plain launch activates its staged tiles, which the emitter does not place.
        ProgramError: If the compiled bundles exceed the IRAM depth limit under either placement.
    """
    if isinstance(launch, FusedLaunch):
        bundles, staging = fused_ffn_stream(chunks=launch.chunks, steps=launch.steps,
                                            output_tiles=launch.output_tiles)
        return _assemble(name="fused_ffn", bundles=bundles,
                         note="Fused launch cannot fit in hardware IRAM"), staging

    if launch.activate:
        message = (
            "Unable to compile a launch activating its staged tiles outside a fused pair. The activation rides "
            "the staging of the matmul consuming the product, which is the fused stream rather than this one."
        )
        raise NotImplementedError(message)

    counts = dict(column_tiles=launch.column_tiles, groups=launch.groups,
                  loop_blocks=launch.loop_blocks,
                  fuse_groups=launch.fuse_groups, block_m=launch.block_m, block_n=launch.block_n)
    bundles, staging = looped_gemm_stream(**counts)
    note = (
        f"Launch cannot fit in hardware IRAM: unrolling {launch.output_tiles} output tiles for a BLOCK_N of "
        f"{launch.block_n} bloats the image, so try a narrower tile width, such as 16, 32 or 64"
    )
    return _assemble(name="looped_gemm", bundles=bundles, note=note), staging
