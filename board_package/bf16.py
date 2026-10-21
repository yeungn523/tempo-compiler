#!/usr/bin/env python3
"""Provides the device geometry, the BF16 packing and the host references every check compares against."""

from __future__ import annotations

import math
import pathlib
import struct
import sys

import numpy as np
from numpy.typing import NDArray

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The directory holding this package."""

# A module that imports bf16 reaches the board runtime through it rather than inserting the roots itself, so
# importing bf16 has to leave both of them on the path.
sys.path.insert(0, str(HERE))

_ELEMENT_BYTES: int = 2
"""The byte width of one BF16 value in memory."""

_NUM_LANES: int = 16
"""The number of MXU lanes, which is how many values one VMEM row holds."""

_ROW_BYTES: int = _NUM_LANES * _ELEMENT_BYTES
"""The byte size of one VMEM row, which is the stride every DMA descriptor walks DRAM at."""

_GELU_TABLE_ENTRIES: int = 2048
"""The number of entries in the SFU's GELU table."""

_GELU_LIMIT_Q7: int = 8 * 128
"""The Q7 magnitude the GELU table saturates at, which is 8.0."""

_CORRUPTION_WORD: int = 0xDEAD_BEEF
"""The sentinel a destination buffer is filled with before a read-back, so a copy that never landed is visible."""

_LAUNCH_REPEATS: int = 3
"""The number of back-to-back launches of one image the repeat check issues."""

_TILE: int = 16
"""The edge length of one MXU tile."""

_TOKEN_GROUPS: int = 2
"""The number of token groups one block carries."""

_STEPS: int = 2
"""The number of accumulation steps one block spans."""

_BLOCK_TOKENS: int = _TOKEN_GROUPS * _TILE
"""The number of token rows one block carries."""

_BLOCK_DEPTH: int = _STEPS * _TILE
"""The contraction depth one block spans."""

_NUM_SUBLANES: int = 4
"""The number of sublanes one vector register spans."""

_BLOCK_ACTIVATION_BYTES: int = _STEPS * _TOKEN_GROUPS * _TILE * _ROW_BYTES
"""The byte size of one block's activation tiles, which the runtime allocates past the activations as slack."""

_BLOCK_WEIGHT_BYTES: int = _STEPS * _TILE * _ROW_BYTES
"""The byte size of one block's weight tiles."""

_ZERO_TILE_WORDS: int = _TILE * _TILE
"""The word count of the zeroed tile that clears the output accumulators."""

_ZERO_TILE_BYTES: int = _ZERO_TILE_WORDS * _ELEMENT_BYTES
"""The byte size of that zeroed tile."""

_VREG_ELEMENTS: int = _NUM_LANES * _NUM_SUBLANES
"""The number of values one vector register holds."""

_MODEL_WIDTH: int = 768
"""The GPT-2 model width in features."""

_LN_EPSILON: float = 1e-5
"""The variance floor LayerNorm adds before taking the reciprocal square root."""


def as_bf16_words(values: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Stores each float32 value in the MINITPU's BF16 container, rounding to nearest even.

    Args:
        values: The values to convert.

    Returns:
        The BF16 container words.
    """
    bits = values.astype("<f4", copy=False).view("<u4")
    words = bits >> 16
    words &= 1
    words += 0x7FFF
    words += bits
    words >>= 16
    return words.astype("<u2")


def from_bf16_words(words: NDArray[np.uint16]) -> NDArray[np.float32]:
    """Recovers the float32 values the MINITPU's BF16 container words hold."""
    return (words.astype("<u4") << 16).view("<f4")


def gelu_lut_reference(words: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Models the SFU's direct GELU table over a vector of BF16 container words.

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


def _padded_columns(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns the array with its last dimension padded with zeros to a whole number of `_TILE`-wide tiles, which is the
    extent an image covers along an axis a kernel masks."""
    short = -values.shape[-1] % _TILE
    if not short:
        return values
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, short)])


def pack_activations(activations: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays an activation array out in the order the looped kernel visits it.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction, the columns padded with
            zeros to whole tiles.

    Returns:
        The BF16 words of every step, ordered contraction block outermost and step innermost.

    Raises:
        ValueError: If the rows are not a whole number of token groups.
    """
    tokens = activations.shape[0]
    if tokens % _TILE:
        message = (
            f"Unable to pack an activation array of shape {list(activations.shape)}. Its rows must be a whole "
            f"number of the {_TILE} a token group holds."
        )
        raise ValueError(message)
    activations = _padded_columns(values=activations)
    depth = activations.shape[1]
    words = as_bf16_words(values=activations)
    return np.ascontiguousarray(words.reshape(tokens, depth // _TILE, _TILE).transpose(1, 0, 2)).reshape(-1)


def pack_bias(bias: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays a bias row out in the order a stride-one broadcast read visits it.

    Args:
        bias: One value per feature, padded with zeros to whole `_NUM_LANES`-wide tiles.

    Returns:
        The BF16 words, each feature tile repeated `_NUM_SUBLANES` times.

    Raises:
        ValueError: If it is not one row.
    """
    if bias.ndim != 1:
        message = f"Unable to pack a bias of shape {bias.shape}. It must be one row."
        raise ValueError(message)
    tiles = as_bf16_words(values=_padded_columns(values=bias)).reshape(-1, _NUM_LANES)
    return np.ascontiguousarray(np.repeat(tiles, _NUM_SUBLANES, axis=0)).reshape(-1)


def pack_weights(weights: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays a weight matrix out in the order the looped kernel visits it.

    Args:
        weights: The right operand, whose rows are the contraction and columns the output features, both padded with
            zeros to whole tiles.

    Returns:
        The BF16 words of every column tile, each tile's contraction rows in order.
    """
    weights = _padded_columns(values=_padded_columns(values=weights).T).T
    depth, columns = weights.shape
    words = as_bf16_words(values=weights)
    ordered = words.reshape(depth, columns // _TILE, _TILE).transpose(1, 0, 2)
    return np.ascontiguousarray(ordered).reshape(-1)


def decode_mask_tiles(*, keys: int, length: int) -> NDArray[np.float32]:
    """Returns the mask tiles a decode step adds to its scores, keeping the first `length` of `keys` cached keys.

    Args:
        keys: Keys the decode image walks, a whole number of tiles.
        length: Keys the cache holds.

    Returns:
        The tiles, shaped [1, key blocks, queries, keys], every query row reading the same keys.
    """
    masked = np.where(np.arange(keys) < length, np.float32(0.0), np.float32(-1.0e30)).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(masked.reshape(1, keys // _TILE, 1, _TILE),
                                                (1, keys // _TILE, _TILE, _TILE)))


def pack_attention_constants(*, initial_maximum: float, constants: tuple[float, ...],
                             masks: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays out the constants block a resident attention image stages for each query block.

    Args:
        initial_maximum: Value the running maximum enters below, which every score clears.
        constants: Numbers the kernel states, a tile of each.
        masks: The tile added to the scores of each pair of a query block and a key block, shaped [query blocks,
            key blocks, queries, keys].

    Returns:
        The words of every query block's initial maximum, zeros, stated numbers and mask tiles, block after block.
    """
    entered = _attention_constants(blocks=masks.shape[0], initial_maximum=initial_maximum, constants=constants,
                                   masks=masks)
    return as_bf16_words(values=np.concatenate(entered, axis=0))


def attention_reference(query: NDArray[np.float32], key: NDArray[np.float32],
                        value: NDArray[np.float32], *, mask: NDArray[np.float32] | None = None) -> NDArray[np.float64]:
    """Returns the attention the block computes, in float64, for a drained image to be held against.

    Args:
        query: The block's queries, whose rows are tokens and columns the head.
        key: Every key the block attends over.
        value: Every value the block attends over.
        mask: The number added to each score, one row per query and one column per key, or None for none.

    Returns:
        The attended output, with one row per query token.
    """
    scores = np.asarray(query, dtype=np.float64) @ np.asarray(key, dtype=np.float64).T
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ np.asarray(value, dtype=np.float64)


def attention_bf16_reference(query: NDArray[np.float32], key: NDArray[np.float32],
                             value: NDArray[np.float32], *, initial_maximum: float,
                             mask: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
    """Returns one attention block computed as the image computes it, rounded at every point it writes a VREG.

    Args:
        query: The block's queries, whose rows are tokens and columns the head.
        key: Every key the block attends over.
        value: Every value the block attends over.
        initial_maximum: Value the running maximum enters below, which the image reads from its operands.
        mask: The number added to each score, one row per query and one column per key, or None for none.

    Returns:
        The attended output, one BF16 container per value throughout.
    """
    def held(values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Returns the values as one BF16 container each, which is what a VREG holds."""
        return from_bf16_words(words=as_bf16_words(values=np.asarray(values, dtype=np.float32)))

    rows, head = query.shape
    running_maximum = np.full((rows, 1), initial_maximum, dtype=np.float32)
    running_sum = np.zeros((rows, 1), dtype=np.float32)
    attended = np.zeros((rows, head), dtype=np.float32)

    for block in range(0, key.shape[0], _TILE):
        keys, values = key[block:block + _TILE], value[block:block + _TILE]
        scores = np.zeros((rows, _TILE), dtype=np.float32)
        for step in range(0, head, _TILE):
            piece = held(query[:, step:step + _TILE] @ keys[:, step:step + _TILE].T)
            scores = piece if step == 0 else held(scores + piece)
        if mask is not None:
            scores = held(scores + mask[:, block:block + _TILE])

        maximum = held(np.maximum(running_maximum, scores.max(axis=1, keepdims=True)))
        alpha = held(np.exp(held(running_maximum - maximum)))
        running_maximum = maximum
        probabilities = held(np.exp(held(scores - maximum)))

        running_sum = held(held(running_sum * alpha) + held(probabilities.sum(axis=1, keepdims=True)))
        attended = held(attended * alpha)
        for tile in range(0, head, _TILE):
            piece = held(probabilities @ values[:, tile:tile + _TILE])
            attended[:, tile:tile + _TILE] = held(attended[:, tile:tile + _TILE] + piece)
    return held(attended * held(1.0 / running_sum))


def assert_operands_discriminating(activations: NDArray[np.float32], weights: NDArray[np.float32]) -> None:
    """Rejects an operand pair whose product cannot observe the confusions a tiled GEMM can make.

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
    candidates = []
    tokens = _reordered(rng, left.shape[0])
    if tokens is not None:
        candidates.append(("tokens", left[tokens] @ right))
    shuffled_blocks = _reordered(rng, blocks)
    if shuffled_blocks is not None:
        by_block = left.reshape(left.shape[0], blocks, _BLOCK_DEPTH)[:, shuffled_blocks, :]
        candidates.append(("contraction blocks", by_block.reshape(left.shape) @ right))
    columns = _reordered(rng, right.shape[1])
    if columns is not None:
        candidates.append(("output columns", left @ right[:, columns]))

    for axis, product in candidates:
        if np.array_equal(product, reference):
            message = (
                f"Unable to use these operands as evidence. Permuting their {axis} leaves the product "
                f"identical, so they cannot distinguish a kernel that confuses one for another from one "
                f"that does not. Vary the operands along every axis the kernel indexes."
            )
            raise ValueError(message)


def build_operands(
    tokens: int, depth: int, columns: int, *, kind: str, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Builds one operand pair for a GEMM check.

    Args:
        tokens: Rows of the left operand.
        depth: The contraction.
        columns: Output features.
        kind: ``structured`` for exactly representable operands, ``varying`` for operands that differ along every
            axis.
        seed: Seed for the varying operands, so two rounds can differ.

    Returns:
        The left and right operands, both on the BF16 grid.

    Raises:
        ValueError: If the varying operands cannot observe the confusions a tiled GEMM can make.
    """
    if kind == "structured":
        left = np.arange(tokens * depth, dtype=np.float32).reshape(tokens, depth) % 2
        scales = (2.0 ** -(np.arange(columns) % 4)) / 16.0
        right = np.broadcast_to(scales, shape=(depth, columns)).astype(np.float32)
        return left, right
    rng = np.random.default_rng(seed)
    left = _on_bf16_grid(values=rng.standard_normal((tokens, depth)).astype(np.float32))
    right = _on_bf16_grid(values=rng.standard_normal((depth, columns)).astype(np.float32))
    # Fails rather than handing back operands that would report a vacuous pass.
    assert_operands_discriminating(activations=left, weights=right)
    return left, right


def _attention_constants(*, blocks: int, initial_maximum: float, constants: tuple[float, ...],
                         masks: NDArray[np.float32] | None) -> list[NDArray[np.float32]]:
    """Returns the tiles each query block enters its key loop from, in the order it stages them."""
    initial = np.full((_TILE, _NUM_LANES), initial_maximum, dtype=np.float32)
    cleared = np.zeros((_TILE, _NUM_LANES), dtype=np.float32)
    stated = [np.full((_TILE, _NUM_LANES), number, dtype=np.float32) for number in constants]
    return [np.concatenate([initial, cleared, *stated, *(() if masks is None else masks[block])], axis=0)
            for block in range(blocks)]


def _reordered(rng: np.random.Generator, count: int) -> NDArray[np.intp] | None:
    """Returns a permutation of the given positions that moves at least one of them.

    An identity draw would leave the operand unchanged, so the check it feeds would read its own no-op as evidence.

    Args:
        rng: The generator the permutation is drawn from.
        count: Positions to permute.

    Returns:
        The permutation, or None when one position cannot be reordered at all.
    """
    if count < 2:
        return None
    identity = np.arange(count)
    order = rng.permutation(count)
    while np.array_equal(order, identity):
        order = rng.permutation(count)
    return order


def _gelu_table() -> NDArray[np.uint16]:
    """Rebuilds the BF16 ROM image the SFU's GELU table is programmed from."""
    centres = -8.0 + (np.arange(_GELU_TABLE_ENTRIES, dtype=np.float64) + 0.5) / 128.0
    values = (0.5 * centre * (1.0 + math.erf(centre / math.sqrt(2.0))) for centre in centres)
    return np.array([_bf16_round(value=value) for value in values], dtype="<u2")


def _on_bf16_grid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rounds values onto the BF16 grid, which is what the device receives.

    Args:
        values: The values to round, of any shape including 0-d.

    Returns:
        The rounded values, in the shape they arrived in.
    """
    array = np.asarray(values, dtype=np.float32)
    return from_bf16_words(words=as_bf16_words(values=array.reshape(-1))).reshape(array.shape)


def _fp32_abs_q7(bits: NDArray[np.uint32]) -> NDArray[np.int64]:
    """Converts FP32 magnitudes to the Q7 integers the SFU builds table addresses from.

    Args:
        bits: FP32 bit patterns with the sign bit already cleared.

    Returns:
        The magnitudes scaled by 128 and truncated, saturated to the 13-bit field the SFU carries.

    Notes:
        The field holds magnitudes below 8192/128, which is 64.0. A 24-bit mantissa reaches 8192 exactly when
        the shift is ten or less and can never reach it when the shift is eleven or more, so the saturation sits
        at that boundary.
    """
    exponent = ((bits >> 23) & 0xFF).astype(np.int64)
    mantissa = (bits & 0x7F_FFFF) | (1 << 23)
    right_shift = 16 - (exponent - 127)
    shifted = mantissa >> np.clip(right_shift, 0, 63)
    return np.where((exponent == 0) | (right_shift >= 24), 0, np.where(right_shift <= 10, 0x1FFF, shifted))


def _bf16_round(value: float) -> int:
    """Rounds one float to round-to-nearest-even BF16 bits, matching the table the SFU is programmed from.

    Args:
        value: The value to round.

    Returns:
        The sixteen BF16 bits.
    """
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
