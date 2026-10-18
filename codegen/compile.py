"""Provides the image one launch compiles to and the operands a host stages for it."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .asm import IRAM_DEPTH, AsmKernel
from bf16 import _BLOCK_DEPTH, _TILE
from .emit import (Staging, flash_attention_stream, looped_gemm_stream, pointwise_stream,
                  resident_attention_stream, row_stream, strided_copy_stream)

from .ir import FlashLaunch, Launch, MatMul, PointwiseLaunch, ProgramError, RowLaunch


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


def compile_launch(launch: Launch | FlashLaunch) -> tuple[bytes, Staging]:
    """Returns the image one launch compiles to and what a host stages for it.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary, and the staging the emitter placed them from.

    Raises:
        ValueError: If an activating launch covers a block wider than one resident output tile.
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
                                                  row_blocks=launch.row_blocks,
                                                  masked=launch.mask is not None)
        # Pins the hand placement, whose gaps carry the VMEM and staging array dependencies the hazard model
        # cannot see, so a packed image would move a load ahead of the store it reads.
        return _assemble(name="flash_attention", bundles=bundles, placements=("",),
                         remedy="narrow the head the block accumulates"), staging

    counts = dict(column_tiles=launch.column_tiles, groups=launch.groups,
                  loop_blocks=launch.loop_blocks, activate=launch.activate,
                  block_m=launch.block_m, block_n=launch.block_n)
    bundles, staging = looped_gemm_stream(**counts)
    remedy = (
        f"narrow the tile width, where a BLOCK_N of {launch.block_n} unrolls {launch.output_tiles} output tiles"
    )
    return _assemble(name="looped_gemm", bundles=bundles, remedy=remedy), staging


def compile_resident_attention(launch: FlashLaunch, *, heads: int, fused_tiles: int,
                               cache_tiles: int | None = None) -> bytes:
    """Returns the image attending every head where the fused QKV tensor lies.

    Args:
        launch: The masked attention launch the planner produced, whose steps the image runs.
        heads: Heads the fused tensor carries.
        fused_tiles: Feature tiles one token of the fused tensor holds.
        cache_tiles: Feature tiles one token of the KV cache holds, for a decode image attending one query block
            over it, or None for an image attending the fused tensor's every token.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If the launch carries no steps or no mask, or the image exceeds the IRAM depth limit.
    """
    if not launch.body or launch.mask is None:
        message = (
            "Unable to compile a resident attention image from a launch carrying no steps or no mask. The image "
            "stages one mask tile per key block, so it runs a kernel selecting its scores by position."
        )
        raise ProgramError(message)
    bundles = resident_attention_stream(key_blocks=launch.key_blocks, head=launch.head, body=launch.body,
                                        epilogue=launch.epilogue, constants=len(launch.constants), heads=heads,
                                        fused_tiles=fused_tiles, cache_tiles=cache_tiles)
    # Pins the hand placement, whose gaps carry the VMEM and staging array dependencies the hazard model cannot see.
    return _assemble(name="resident_attention", bundles=bundles, placements=("",),
                     remedy="narrow the head the block accumulates")


def compile_copy(*, rows: int, stride: int) -> bytes:
    """Returns the image copying rows spaced `stride` bytes apart from its first argument to its third.

    Args:
        rows: Rows to copy, each 32 bytes.
        stride: Bytes between one row and the next at both ends.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If the compiled bundles exceed the IRAM depth limit.
    """
    return _assemble(name="copy", bundles=strided_copy_stream(rows=rows, stride=stride), placements=("",),
                     remedy="copy fewer rows per launch")


def compile_pointwise(launch: PointwiseLaunch) -> bytes:
    """Returns the image one pointwise launch compiles to.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If the compiled bundles exceed the IRAM depth limit.
    """
    bundles = pointwise_stream(chunks=launch.chunks, broadcast_right=launch.broadcast_right, body=launch.body)
    # Pins the hand placement, whose gaps carry the VMEM dependencies the hazard model cannot see.
    return _assemble(name="pointwise", bundles=bundles, placements=("",), remedy="shorten the arithmetic it runs")


def compile_rows(launch: RowLaunch) -> bytes:
    """Returns the image one row launch compiles to.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary.

    Raises:
        ProgramError: If the compiled bundles exceed the IRAM depth limit.
    """
    bundles = row_stream(features=launch.features, rows=launch.rows, scalars=len(launch.scalars),
                         prologue=launch.prologue, body=launch.body)
    # Packs because the image waits on a dependent chain of reductions, which co-issuing the work between shortens.
    return _assemble(name="rows", bundles=bundles, placements=("pack",), remedy="sum fewer values over the row")


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
