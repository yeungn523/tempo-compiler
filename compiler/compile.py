"""Provides the image one launch compiles to and the operands a host stages for it."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asm import IRAM_DEPTH, AsmKernel
from bf16 import _BLOCK_DEPTH, _TILE
from emit import Staging, flash_attention_stream, fused_ffn_stream, looped_gemm_stream

from .ir import FlashLaunch, FusedLaunch, Launch, MatMul, ProgramError


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


def compile_launch(launch: Launch | FusedLaunch | FlashLaunch) -> tuple[bytes, Staging]:
    """Returns the image one launch compiles to and what a host stages for it.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary, and the staging the emitter placed them from.

    Raises:
        NotImplementedError: If a plain launch activates its staged tiles, which the emitter does not place.
        ProgramError: If the compiled bundles exceed the IRAM depth limit under every placement tried.
    """
    if isinstance(launch, FlashLaunch):
        if not launch.body or not launch.epilogue:
            message = (
                "Unable to compile an attention launch carrying no steps. The steps one key block runs are "
                "placed from the kernel's own operations, so a launch stating its extents alone is one no "
                "plan made."
            )
            raise ProgramError(message)
        bundles, staging = flash_attention_stream(key_blocks=launch.key_blocks, queries=launch.queries,
                                                  head=launch.head, body=launch.body,
                                                  epilogue=launch.epilogue,
                                                  constants=len(launch.constants),
                                                  row_blocks=launch.row_blocks)
        # Pins the hand placement, whose gaps carry the VMEM and staging array dependencies the hazard model
        # cannot see, so a packed image would move a load ahead of the store it reads.
        return _assemble(name="flash_attention", bundles=bundles, placements=("",),
                         remedy="narrow the head the block accumulates"), staging

    if isinstance(launch, FusedLaunch):
        bundles, staging = fused_ffn_stream(chunks=launch.chunks, steps=launch.steps,
                                            output_tiles=launch.output_tiles)
        return _assemble(name="fused_ffn", bundles=bundles,
                         remedy="narrow the output the layer covers"), staging

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
    remedy = (
        f"narrow the tile width, where a BLOCK_N of {launch.block_n} unrolls {launch.output_tiles} output tiles"
    )
    return _assemble(name="looped_gemm", bundles=bundles, remedy=remedy), staging


def _assemble(*, name: str, bundles: tuple[int, ...], remedy: str,
              placements: tuple[str | None, ...] = (None, "pack")) -> bytes:
    """Returns one kernel's image, packing its bundles where the hand placement overflows IRAM.

    Args:
        name: The name the kernel's error messages carry.
        bundles: The bundle words the emitter placed, in program order.
        remedy: What a refusal asks the caller to narrow, the tile that fits being the kernel's own choice.
        placements: Re-placement passes tried in turn, an image whose schedule carries its own dependencies
            naming the one pass that leaves it as it was written.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If no placement fits the image in IRAM.
    """
    overflow: ValueError | None = None
    for placement in placements:
        try:
            return AsmKernel(name=name, bundles=bundles, placement=placement).to_bytes(pad_to=4)
        except ValueError as error:
            overflow = error
    message = (
        f"Unable to fit the {name} image in IRAM. Its {len(bundles)} bundles exceed the {IRAM_DEPTH} the memory "
        f"holds under the hand placement and under packing alike, so {remedy}. {overflow}"
    )
    raise ProgramError(message) from overflow
