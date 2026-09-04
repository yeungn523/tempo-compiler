#!/usr/bin/env python3
"""Device geometry, BF16 packing and the host references every check compares against.

Nothing here touches the device. It is the half of the package a reviewer can read and check by
inspection: how a float32 becomes the BF16 word the array consumes, how an operand is laid out for
the kernel that reads it, and what the SFU's own tables do -- all derived from the geometry
constants at the top rather than written out, because every one of them silently mis-addresses
rather than failing when the device geometry moves under it.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

import math  # noqa: E402
import struct  # noqa: E402

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent


# --- Device geometry ------------------------------------------------------
# Derived rather than written out, because every one of these silently
# mis-addresses rather than failing when the device geometry moves under it.
_ELEMENT_BYTES: int = 2


_NUM_LANES: int = 16


_ROW_BYTES: int = _NUM_LANES * _ELEMENT_BYTES


_VMEM_ENTRIES_PER_LANE: int = 8192


_GELU_TABLE_ENTRIES: int = 2048


_GELU_LIMIT_Q7: int = 8 * 128


_CORRUPTION_WORD: int = 0xDEAD_BEEF


_LAUNCH_REPEATS: int = 3


_INTERLEAVE_ROUNDS: int = 4


_INTERLEAVE_PAYLOAD_BYTES: int = 9216


_TILE: int = 16


_TOKEN_GROUPS: int = 2


_STEPS: int = 2


_BLOCK_TOKENS: int = _TOKEN_GROUPS * _TILE


_BLOCK_DEPTH: int = _STEPS * _TILE


_STEP_TILES: int = 4


_LOOP_BLOCKS: int = 6


_NUM_SUBLANES: int = 4


_TILE_VREGS: int = 4


_TILE_ROWS: int = _NUM_SUBLANES * _TILE_VREGS


_ROW_ZERO: int = 0


_ROW_ACTIVATION: tuple[int, int] = (16, 80)


_ROW_WEIGHT: tuple[int, int] = (64, 128)


_ROW_OUTPUT: int = 160


_VMEM_SLOT_STRIDE: int = _VMEM_ENTRIES_PER_LANE


_BLOCK_ACTIVATION_BYTES: int = _STEPS * _TOKEN_GROUPS * _TILE * _ROW_BYTES


_BLOCK_WEIGHT_BYTES: int = _STEPS * _TILE * _ROW_BYTES


_ZERO_TILE_WORDS: int = _TILE * _TILE


_ZERO_TILE_BYTES: int = _ZERO_TILE_WORDS * _ELEMENT_BYTES


_ROW_SCORES: int = 256


_ROW_SOFTMAX_OUT: int = 1024


_VREG_ELEMENTS: int = _NUM_LANES * _NUM_SUBLANES


_MAX_DESCRIPTOR_ROWS: int = 128


#: VMEM rows the elementwise add stages its two operands and its result at. One
#: chunk is _MAX_DESCRIPTOR_ROWS rows, which is what a single descriptor moves.
_ROW_ADD_LEFT: int = 2048


_ROW_ADD_RIGHT: int = _ROW_ADD_LEFT + _MAX_DESCRIPTOR_ROWS


_ROW_ADD_OUT: int = _ROW_ADD_RIGHT + _MAX_DESCRIPTOR_ROWS


#: VMEM rows the tile transpose stages its input and its result at. One tile is
#: _TILE rows in and _TILE rows out.
_ROW_TX_IN: int = _ROW_ADD_OUT + _MAX_DESCRIPTOR_ROWS


_ROW_TX_OUT: int = _ROW_TX_IN + _TILE


_MODEL_WIDTH: int = 768


_WIDTH_VREGS: int = _MODEL_WIDTH // _VREG_ELEMENTS


_WIDTH_ROWS: int = _MODEL_WIDTH // _NUM_LANES


_LN_ROW_X: int = 0


_LN_ROW_CONST: int = _WIDTH_ROWS


_LN_ROW_OUT: int = 512


_LN_EPSILON: float = 1e-5


def layernorm_constants(gamma: NDArray[np.float32], beta: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Packs the resident half of a LayerNorm launch: gamma, beta and three broadcast scalars."""
    zero = np.zeros(_VREG_ELEMENTS, dtype=np.float32)
    inverse = np.full(_VREG_ELEMENTS, 1.0 / _MODEL_WIDTH, dtype=np.float32)
    epsilon = np.full(_VREG_ELEMENTS, _LN_EPSILON, dtype=np.float32)
    return as_bf16_words(values=np.concatenate([gamma, beta, zero, inverse, epsilon]))


def as_bf16_words(values: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Stores each float32 value in the MINITPU's 32-bit BF16 container format.

    Notes:
        Rounds to nearest even, matching the MXU's pack_bf16, the accumulator's add pipe and the SFU's table, so the
        host is the same rounding mode as every stage it feeds.
    """
    bits = values.astype("<f4", copy=False).view("<u4")
    words = bits >> 16
    words &= 1
    words += 0x7FFF
    words += bits
    words >>= 16
    return words.astype("<u2")


def from_bf16_words(words: NDArray[np.uint32]) -> NDArray[np.float32]:
    """Recovers the float32 values the MINITPU's 32-bit BF16 container words hold."""
    return (words.astype("<u4") << 16).view("<f4")


def pack_gemm_input(activations: NDArray[np.float32], weights: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Packs the tiled GEMM's tiles into the one buffer its DMA prologue expects.

    Notes:
        Orders the blocks to match the kernel's VMEM bank map: a zeroed block clearing the output tiles, then one
        block per accumulation step holding that step's activation tiles followed by its weight tile. One tile
        occupies sixteen VMEM rows.

        A step block always spans _STEP_TILES tiles, since the kernel's AGU walks the steps with a power-of-two row
        stride and its weight bank sits at the top of that stride. Fewer activation tiles than the block holds
        therefore leaves a gap ahead of the weight tile, which this padding fills.

    Args:
        activations: Activation tiles indexed by step, group, row and column.
        weights: Weight tiles indexed by step, row and column.

    Returns:
        The VMEM image as BF16 container words, spanning one clearing tile plus _STEP_TILES tiles per step.

    Raises:
        ValueError: If a step holds more activation tiles than the AGU's stride leaves room for.
    """
    steps, groups = activations.shape[0], activations.shape[1]
    if groups > _STEP_TILES - 1:
        message = (
            f"Unable to pack the GEMM image for {groups} token groups. A step block spans {_STEP_TILES} tiles and "
            f"reserves the last for the weights, so the group count must be at most {_STEP_TILES - 1}."
        )
        raise ValueError(message)

    padding = np.zeros_like(weights[0])
    blocks = [padding]
    for step in range(steps):
        blocks.extend(activations[step, group] for group in range(groups))
        blocks.extend(padding for _ in range(_STEP_TILES - 1 - groups))
        blocks.append(weights[step])
    return np.concatenate([as_bf16_words(values=block).ravel() for block in blocks])


def gelu_lut_reference(words: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Models the SFU's direct GELU table over a vector of BF16 container words.

    Notes:
        Mirrors the address fold in src/core/sfu/sfu.sv. The operand magnitude becomes a Q7 integer, the sign
        selects one half of the table, and the entry holds that bin's centre GELU value rounded to BF16. Magnitudes
        at or past the limit pass through when the operand is positive and return zero when it is negative.

    Args:
        words: BF16 container words to transform.

    Returns:
        The container words the SFU produces for them.
    """
    operand = words.astype("<u4") << 16
    negative = (operand >> 31).astype(bool)
    magnitude = _fp32_abs_q7(bits=operand & 0x7FFF_FFFF)
    address = np.where(negative, _GELU_LIMIT_Q7 - 1 - magnitude, _GELU_LIMIT_Q7 + magnitude)
    entry = _gelu_table()[np.clip(address, 0, _GELU_TABLE_ENTRIES - 1)]
    saturated = np.where(negative, 0, words.astype("<u4"))
    result = np.where(magnitude >= _GELU_LIMIT_Q7, saturated, entry)
    return np.where((operand & 0x7FFF_FFFF) == 0, 0, result).astype("<u4")


def pack_activations(activations: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Lays an activation array out in the order the looped kernel visits it.

    Notes:
        Packing both operands gives every descriptor the same 64-byte row pitch, which frees the register that
        would otherwise hold the array's own pitch so the drain pointer can live across the group loop.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction.

    Returns:
        The BF16 words of every step, ordered contraction block outermost and step innermost.

    Raises:
        ValueError: If a dimension is not a whole number of blocks.
    """
    tokens, depth = activations.shape
    if tokens % _BLOCK_TOKENS or depth % _BLOCK_DEPTH:
        message = (
            f"Unable to pack an activation array of shape [{tokens},{depth}]. Its rows must be a whole number "
            f"of {_BLOCK_TOKENS} and its columns a whole number of {_BLOCK_DEPTH}."
        )
        raise ValueError(message)
    words = as_bf16_words(values=activations)
    return np.ascontiguousarray(words.reshape(tokens, depth // _TILE, _TILE).transpose(1, 0, 2)).reshape(-1)


def pack_weights(weights: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Lays a weight matrix out in the order the looped kernel visits it.

    Notes:
        Consecutive blocks have to be consecutive in memory, because the kernel advances its weight pointer by
        one scalar add per step. A natural row-major block advances 196608 bytes at model width, past both the
        immediate an add carries and the multiplier a multiply-accumulate carries.

    Args:
        weights: The right operand, whose rows are the contraction and columns the output features.

    Returns:
        The BF16 words of every block, ordered output tile outermost and contraction block innermost.

    Raises:
        ValueError: If a dimension is not a whole number of blocks.
    """
    depth, columns = weights.shape
    if depth % _BLOCK_DEPTH or columns % _TILE:
        message = (
            f"Unable to pack a weight matrix of shape [{depth},{columns}]. Both dimensions must be a whole "
            f"number of the {_BLOCK_DEPTH}x{_TILE} block one group stages."
        )
        raise ValueError(message)
    words = as_bf16_words(values=weights)
    ordered = words.reshape(depth // _BLOCK_DEPTH, _BLOCK_DEPTH, columns // _TILE, _TILE).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(ordered).reshape(-1)


def assert_operands_discriminating(
    activations: NDArray[np.float32], weights: NDArray[np.float32]
) -> None:
    """Rejects an operand pair whose product cannot observe the confusions a tiled GEMM can make.

    Notes:
        A test is only evidence for the axes its operands vary along. The check that stood for weeks
        used a left operand whose token rows were all identical (``arange % 2`` over an even depth)
        against weights constant along the contraction. Every 32-deep block then contributes the same
        value, so a kernel that pairs the wrong activation block with the wrong weight block -- or
        drops a block's results entirely -- returns exactly the same answer, bit for bit, at
        ``rtol=0, atol=0``. That reads as the strongest possible evidence and is none at all.

        Each permutation below models one such confusion. If the product does not move under it, the
        operands cannot tell a kernel that makes it from one that does not.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction.
        weights: The right operand, whose rows are the contraction and columns the output features.

    Raises:
        ValueError: If the product is unchanged under any of the three permutations.
    """
    left = activations.astype(np.float64)
    right = weights.astype(np.float64)
    reference = left @ right
    rng = np.random.default_rng(0)

    blocks = left.shape[1] // _BLOCK_DEPTH
    shuffled_blocks = rng.permutation(blocks)
    by_block = left.reshape(left.shape[0], blocks, _BLOCK_DEPTH)[:, shuffled_blocks, :]

    for axis, product in (
        ("tokens", left[rng.permutation(left.shape[0])] @ right),
        ("contraction blocks", by_block.reshape(left.shape) @ right),
        ("output columns", left @ right[:, rng.permutation(right.shape[1])]),
    ):
        if np.array_equal(product, reference):
            message = (
                f"Unable to use these operands as evidence. Permuting their {axis} leaves the product "
                f"identical, so they cannot distinguish a kernel that confuses one for another from one "
                f"that does not. Vary the operands along every axis the kernel indexes."
            )
            raise ValueError(message)


def _single_launch_operands() -> tuple[NDArray[np.float32], NDArray[np.float32],
                                       NDArray[np.float32], NDArray[np.uint32]]:
    """Builds the operands every single-launch check shares.

    Notes:
        Each activation row holds one more leading one than the row above it, and the count restarts at a
        different place per group and per step. Every row therefore sums to a different value, so a kernel that
        transposed a tile would return a different product and fail the checks that use these.

    Returns:
        The activation tiles, the weight tiles, the product a host reference computes from them, and the packed
        VMEM image one launch takes.
    """
    activations = np.zeros((_STEPS, _TOKEN_GROUPS, _TILE, _TILE), dtype=np.float32)
    for step in range(_STEPS):
        for group in range(_TOKEN_GROUPS):
            for row in range(_TILE):
                activations[step, group, row, : (row + 4 * group + 8 * step) % _TILE + 1] = 1.0
    weights = np.full((_STEPS, _TILE, _TILE), 1.0 / 16.0, dtype=np.float32)
    expected = np.stack(
        [sum(activations[step, group] @ weights[step] for step in range(_STEPS))
         for group in range(_TOKEN_GROUPS)]
    )
    return activations, weights, expected, pack_gemm_input(activations=activations, weights=weights)


def _gelu_table() -> NDArray[np.uint32]:
    """Rebuilds the BF16 ROM image that tools/emit_sfu_tables.py writes for the SFU."""
    centres = -8.0 + (np.arange(_GELU_TABLE_ENTRIES, dtype=np.float64) + 0.5) / 128.0
    values = (0.5 * centre * (1.0 + math.erf(centre / math.sqrt(2.0))) for centre in centres)
    return np.array([_bf16_round(value=value) for value in values], dtype="<u2")


def _on_bf16_grid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns the values the device receives, which is the BF16 grid the host packing rounds them onto.

    Notes:
        Shape-preserving including 0-d, because a reference computes with scalars as well as arrays
        and numpy cannot re-view a 0-d array at a different itemsize.
    """
    array = np.asarray(values, dtype=np.float32)
    return from_bf16_words(words=as_bf16_words(values=array.reshape(-1))).reshape(array.shape)


def _fp32_abs_q7(bits: NDArray[np.uint32]) -> NDArray[np.int64]:
    """Converts FP32 magnitudes to the Q7 integers the SFU builds table addresses from.

    Args:
        bits: FP32 bit patterns with the sign bit already cleared.

    Returns:
        The magnitudes scaled by 128 and truncated, saturated to the 13-bit field the SFU carries.
    """
    exponent = ((bits >> 23) & 0xFF).astype(np.int64)
    mantissa = (bits & 0x7F_FFFF) | (1 << 23)
    right_shift = 16 - (exponent - 127)
    shifted = mantissa >> np.clip(right_shift, 0, 63)
    return np.where((exponent == 0) | (right_shift >= 24), 0, np.where(right_shift <= 0, 0x1FFF, shifted))


def _bf16_round(value: float) -> int:
    """Rounds one float to round-to-nearest-even BF16 bits, matching tools/emit_sfu_tables.py.

    Args:
        value: The value to round.

    Returns:
        The 16 BF16 bits.
    """
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
