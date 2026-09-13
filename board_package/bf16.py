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

# dsl.py reaches the board runtime through this module rather than inserting the roots itself, so importing bf16
# has to leave both of them on the path.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

_ELEMENT_BYTES: int = 2
"""The byte width of one BF16 value in memory."""

_NUM_LANES: int = 16
"""The number of MXU lanes, which is how many values one VMEM row holds."""

_ROW_BYTES: int = _NUM_LANES * _ELEMENT_BYTES
"""The byte size of one VMEM row, which is the stride every DMA descriptor walks DRAM at."""

_VMEM_ENTRIES_PER_LANE: int = 8192
"""The number of rows one VMEM lane holds."""

_GELU_TABLE_ENTRIES: int = 2048
"""The number of entries in the SFU's GELU table."""

_GELU_LIMIT_Q7: int = 8 * 128
"""The Q7 magnitude the GELU table saturates at, which is 8.0."""

_CORRUPTION_WORD: int = 0xDEAD_BEEF
"""The sentinel a destination buffer is filled with before a read-back, so a copy that never landed is visible."""

_LAUNCH_REPEATS: int = 3
"""The number of back-to-back launches of one image the repeat check issues."""

_INTERLEAVE_ROUNDS: int = 4
"""The number of alternating copy and launch rounds the command interleave check issues."""

_INTERLEAVE_PAYLOAD_BYTES: int = 9216
"""The byte size of the payload each interleave round moves."""

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

_STEP_TILES: int = 4
"""The number of tiles one step block spans, set by the AGU's power-of-two row stride."""

_LOOP_BLOCKS: int = 6
"""The number of blocks the looped GEMM accumulates before it drains."""

_NUM_SUBLANES: int = 4
"""The number of sublanes one vector register spans."""

_TILE_VREGS: int = 4
"""The number of vector registers one tile occupies."""

_TILE_ROWS: int = _NUM_SUBLANES * _TILE_VREGS
"""The number of VMEM rows one tile occupies."""

_ROW_ZERO: int = 0
"""The VMEM row of the zeroed tile that clears the output accumulators."""

_ROW_ACTIVATION: tuple[int, int] = (16, 80)
"""The first VMEM row of each accumulation step's activation bank."""

_ROW_WEIGHT: tuple[int, int] = (64, 128)
"""The first VMEM row of each accumulation step's weight bank."""

_ROW_OUTPUT: int = 160
"""The first VMEM row of the output tiles."""

_VMEM_SLOT_STRIDE: int = _VMEM_ENTRIES_PER_LANE
"""The VMEM row distance between consecutive staging slots, which is a whole lane so no two slots share a row."""

_BLOCK_ACTIVATION_BYTES: int = _STEPS * _TOKEN_GROUPS * _TILE * _ROW_BYTES
"""The byte size of one block's activation tiles, at the token block the hand-written kernels cover. An emitted
image takes this from `emit.Staging.activation_slack` instead, which reads the rows its own block covers."""

_BLOCK_WEIGHT_BYTES: int = _STEPS * _TILE * _ROW_BYTES
"""The byte size of one block's weight tiles."""

_ZERO_TILE_WORDS: int = _TILE * _TILE
"""The word count of the zeroed tile that clears the output accumulators."""

_ZERO_TILE_BYTES: int = _ZERO_TILE_WORDS * _ELEMENT_BYTES
"""The byte size of that zeroed tile."""

_ROW_SCORES: int = 256
"""The first VMEM row of the attention scores the softmax reads."""

_ROW_SOFTMAX_OUT: int = 1024
"""The first VMEM row of the softmax result."""

_VREG_ELEMENTS: int = _NUM_LANES * _NUM_SUBLANES
"""The number of values one vector register holds."""

def _max_attention_rows() -> int:
    """Returns the rows one attention launch covers, bounded by the VMEM the scores are staged in.

    Returns:
        The rows, a whole number of VREG spans.

    Raises:
        ValueError: If the gap cannot hold one row per token of a block.
    """
    per_token = (_ROW_SOFTMAX_OUT - _ROW_SCORES) // _BLOCK_TOKENS
    if per_token < 1:
        message = (
            f"Unable to stage {_BLOCK_TOKENS} tokens of scores in {_ROW_SOFTMAX_OUT - _ROW_SCORES} rows. Move "
            f"`_ROW_SOFTMAX_OUT`."
        )
        raise ValueError(message)
    rows_per_token = 1 << (per_token.bit_length() - 1)
    return rows_per_token * _TILE // _VREG_ELEMENTS * _VREG_ELEMENTS


MAX_ATTENTION_ROWS: int = _max_attention_rows()
"""Rows one attention launch covers, which the score staging bounds rather than the VREG a reduction spans."""


_MAX_DESCRIPTOR_ROWS: int = 128
"""The largest number of rows one DMA descriptor moves, which is the chunk every staged region is sized in."""

_ROW_ADD_LEFT: int = 2048
"""The first VMEM row of the elementwise add's left operand."""

_ROW_ADD_RIGHT: int = _ROW_ADD_LEFT + _MAX_DESCRIPTOR_ROWS
"""The first VMEM row of the elementwise add's right operand."""

_ROW_ADD_OUT: int = _ROW_ADD_RIGHT + _MAX_DESCRIPTOR_ROWS
"""The first VMEM row of the elementwise add's result."""

_ROW_TX_IN: int = _ROW_ADD_OUT + _MAX_DESCRIPTOR_ROWS
"""The first VMEM row of the tile transpose's input."""

_ROW_TX_OUT: int = _ROW_TX_IN + _TILE
"""The first VMEM row of the tile transpose's result."""

_ROW_ADD_BIAS: int = _ROW_TX_OUT + _TILE
"""First VMEM row of the replicated bias, which the broadcasting add reads at stride one."""

_MODEL_WIDTH: int = 768
"""The GPT-2 model width in features."""

_WIDTH_VREGS: int = _MODEL_WIDTH // _VREG_ELEMENTS
"""The number of vector registers one model-width row spans."""

_WIDTH_ROWS: int = _MODEL_WIDTH // _NUM_LANES
"""The number of VMEM rows one model-width row spans."""

_LN_ROW_X: int = 0
"""The first VMEM row of the LayerNorm input."""

_LN_ROW_CONST: int = _WIDTH_ROWS
"""The first VMEM row of the LayerNorm constants, which sit directly above the input."""

_LN_ROW_OUT: int = 512
"""The first VMEM row of the LayerNorm result."""

_LN_EPSILON: float = 1e-5
"""The variance floor LayerNorm adds before taking the reciprocal square root."""


def layernorm_constants(gamma: NDArray[np.float32], beta: NDArray[np.float32], *,
                        width: int | None = None) -> NDArray[np.uint16]:
    """Packs the resident half of a LayerNorm launch: gamma, beta and three broadcast scalars.

    Args:
        gamma: The per-feature scale, `width` long.
        beta: The per-feature shift, `width` long.
        width: The model width the mean divides by, defaulting to GPT-2's. Its reciprocal is baked in here, which
            nothing downstream can tell disagrees with the row it is applied to.

    Returns:
        The container words the launch stages in VMEM.

    Raises:
        ValueError: If the operands are not one row of `width` features each.
    """
    width = _MODEL_WIDTH if width is None else width
    _require_norm_operands(width=width, gamma=gamma, beta=beta)
    zero = np.zeros(_VREG_ELEMENTS, dtype=np.float32)
    inverse = np.full(_VREG_ELEMENTS, 1.0 / width, dtype=np.float32)
    epsilon = np.full(_VREG_ELEMENTS, _LN_EPSILON, dtype=np.float32)
    return as_bf16_words(values=np.concatenate([gamma, beta, zero, inverse, epsilon]))


def _require_norm_operands(*, width: int, **operands: NDArray[np.float32]) -> None:
    """Rejects a normalisation operand that is not exactly one row of `width` features.

    Args:
        width: Features one normalised row covers.
        operands: The per-feature vectors to check, named for the message.

    Raises:
        ValueError: If the width is not a whole number of VREGs, or an operand disagrees with it.
    """
    if width <= 0 or width % _VREG_ELEMENTS:
        message = (
            f"Unable to normalise a row of {width} features. The reductions are per VREG, so a row is a whole "
            f"number of {_VREG_ELEMENTS}."
        )
        raise ValueError(message)
    for name, values in operands.items():
        if values.ndim != 1 or values.size != width:
            message = (
                f"Unable to use {name} of shape {values.shape} at width {width}. It must be exactly one row of "
                f"{width} features."
            )
            raise ValueError(message)


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


def reduce_drain(partials: NDArray[np.float32], *, shape: tuple[int, int, int, int, int]) -> NDArray[np.float32]:
    """Sums one launch's drained partials into the token block's product.

    The drain lies in the order the image walked its loops, so a block's output tiles sit inside its partials.

    Args:
        partials: The drained partials, with the clearing tile already skipped.
        shape: Extent of each axis of the drain, as the plan's `Drain` states it.

    Returns:
        The product, as token rows by the columns the launch covers.
    """
    blocks, _, output_tiles, rows, tile = shape
    tiles = blocks * output_tiles
    summed = partials.reshape(shape).sum(axis=1, dtype=np.float32)
    return summed.reshape(tiles, rows, tile).transpose(1, 0, 2).reshape(rows, tiles * tile)


def pack_gemm_input(activations: NDArray[np.float32], weights: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Packs the tiled GEMM's tiles into the one buffer its DMA prologue expects.

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


def pack_activations(activations: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays an activation array out in the order the looped kernel visits it.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction.

    Returns:
        The BF16 words of every step, ordered contraction block outermost and step innermost.

    Raises:
        ValueError: If a dimension is not a whole number of blocks.
    """
    tokens, depth = activations.shape
    if tokens % _TILE or depth % _BLOCK_DEPTH:
        message = (
            f"Unable to pack an activation array of shape [{tokens},{depth}]. Its rows must be a whole number "
            f"of the {_TILE} a token group holds and its columns a whole number of {_BLOCK_DEPTH}."
        )
        raise ValueError(message)
    words = as_bf16_words(values=activations)
    return np.ascontiguousarray(words.reshape(tokens, depth // _TILE, _TILE).transpose(1, 0, 2)).reshape(-1)


def pack_bias(bias: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays a bias row out in the order a stride-one broadcast read visits it.

    Args:
        bias: One value per feature, a whole number of `_NUM_LANES`-wide tiles.

    Returns:
        The BF16 words, each feature tile repeated `_NUM_SUBLANES` times.

    Raises:
        ValueError: If it is not one row of whole tiles.
    """
    if bias.ndim != 1 or bias.size % _NUM_LANES:
        message = (
            f"Unable to pack a bias of shape {bias.shape}. It must be one row of whole {_NUM_LANES}-lane tiles."
        )
        raise ValueError(message)
    tiles = as_bf16_words(values=bias).reshape(-1, _NUM_LANES)
    return np.ascontiguousarray(np.repeat(tiles, _NUM_SUBLANES, axis=0)).reshape(-1)


def pack_weights(weights: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Lays a weight matrix out in the order the looped kernel visits it.

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

def pack_fused_ffn(activations: NDArray[np.float32], up_weight: NDArray[np.float32],
                   down_weight: NDArray[np.float32]) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """Lays the fused feed-forward layer's operands out in the order its kernel visits them.

    Both weights ride one buffer because the kernel holds three pointers in the four scalar registers the machine
    has, so a chunk's up projection is followed by the down projection rows that same chunk contracts.

    Args:
        activations: The layer input, whose rows are tokens and columns the up projection's contraction.
        up_weight: The up projection, whose rows are that contraction and columns the hidden width.
        down_weight: The down projection, whose rows are the hidden width and columns the layer's output.

    Returns:
        The input words, contraction tile outermost, and the weight words, one hidden chunk at a time with
        that chunk's down projection laid out tile by tile.

    Raises:
        ValueError: If a dimension is not a whole number of tiles, if the two weights disagree on the hidden
            width, or if the shape is not the one block the kernel covers.
    """
    tokens, depth = activations.shape
    if tokens != _BLOCK_TOKENS or depth % _TILE:
        message = (
            f"Unable to pack an input of shape [{tokens},{depth}]. One launch covers {_BLOCK_TOKENS} token rows "
            f"and a whole number of {_TILE} wide contraction tiles."
        )
        raise ValueError(message)
    if up_weight.shape[0] != depth:
        message = (
            f"Unable to contract a [{tokens},{depth}] input against an up projection of shape "
            f"{up_weight.shape}. Its rows are the contraction the input's columns walk."
        )
        raise ValueError(message)

    hidden = up_weight.shape[1]
    if hidden % _TILE or down_weight.shape[0] != hidden:
        message = (
            f"Unable to pass a hidden width of {hidden} to a down projection of shape {down_weight.shape}. The "
            f"width must be a whole number of {_TILE} and the two projections must state the same one."
        )
        raise ValueError(message)
    if down_weight.shape[1] % _TILE:
        message = (
            f"Unable to drain a down projection {down_weight.shape[1]} columns wide. The layer sweeps its "
            f"output a {_TILE} wide tile at a time, so its width is a whole number of them."
        )
        raise ValueError(message)

    # Walks the contraction tile by tile, each one every token row of that tile's columns, which is the order the
    # kernel's loads step through after one descriptor has staged the whole input.
    words = as_bf16_words(values=activations)
    staged = np.ascontiguousarray(words.reshape(tokens, depth // _TILE, _TILE).transpose(1, 0, 2)).reshape(-1)

    # Lays each chunk's down projection out tile by tile, which is the order its sweep loads them in.
    tiles = down_weight.shape[1] // _TILE
    chunks = []
    for first in range(0, hidden, _TILE):
        rows = as_bf16_words(values=down_weight[first:first + _TILE])
        swept = np.ascontiguousarray(rows.reshape(_TILE, tiles, _TILE).transpose(1, 0, 2)).reshape(-1)
        chunks.append(np.concatenate((as_bf16_words(values=up_weight[:, first:first + _TILE]).reshape(-1),
                                      swept)))
    return staged, np.concatenate(chunks)


def fused_ffn_reference(activations: NDArray[np.float32], up_weight: NDArray[np.float32],
                        down_weight: NDArray[np.float32]) -> NDArray[np.float64]:
    """Returns the layer the fused kernel computes, in float64, for a drained image to be held against.

    Args:
        activations: The layer input, whose rows are tokens.
        up_weight: The up projection, contracted against the input's columns.
        down_weight: The down projection, contracted against the hidden width.

    Returns:
        The layer output, with one row per input token.
    """
    hidden = np.asarray(activations, dtype=np.float64) @ np.asarray(up_weight, dtype=np.float64)
    activated = 0.5 * hidden * (1.0 + _erf(hidden / math.sqrt(2.0)))
    return activated @ np.asarray(down_weight, dtype=np.float64)


def fused_ffn_bf16_reference(activations: NDArray[np.float32], up_weight: NDArray[np.float32],
                             down_weight: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns the layer computed as the kernel computes it, rounded at every point it writes a VREG.

    Holding a drained layer against this separates what the image got wrong from what BF16 costs over the
    contraction, one number against the exact reference leaving the two indistinguishable.

    Args:
        activations: The layer input, whose rows are tokens.
        up_weight: The up projection, contracted against the input's columns.
        down_weight: The down projection, contracted against the hidden width.

    Returns:
        The layer output, one BF16 container per value throughout.
    """
    def held(values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Returns the values as one BF16 container each, which is what a VREG holds."""
        return from_bf16_words(words=as_bf16_words(values=np.asarray(values, dtype=np.float32)))

    tokens, depth = activations.shape
    out = np.zeros((tokens, down_weight.shape[1]), dtype=np.float32)
    for chunk in range(0, up_weight.shape[1], _TILE):
        hidden = np.zeros((tokens, _TILE), dtype=np.float32)
        for step in range(0, depth, _TILE):
            hidden = held(hidden + held(activations[:, step:step + _TILE]
                                        @ up_weight[step:step + _TILE, chunk:chunk + _TILE]))
        hidden = held(0.5 * hidden * (1.0 + _erf(hidden / math.sqrt(2.0))))
        for tile in range(0, down_weight.shape[1], _TILE):
            piece = held(hidden @ down_weight[chunk:chunk + _TILE, tile:tile + _TILE])
            out[:, tile:tile + _TILE] = held(out[:, tile:tile + _TILE] + piece)
    return out


def _erf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Returns the error function of each value, which the exact GELU is written in."""
    return np.vectorize(math.erf, otypes=[np.float64])(values)


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


def _single_launch_operands() -> (
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.uint16]]
):
    """Builds the operands every single-launch check shares.

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
        [sum(activations[step, group] @ weights[step] for step in range(_STEPS)) for group in range(_TOKEN_GROUPS)]
    )
    return activations, weights, expected, pack_gemm_input(activations=activations, weights=weights)


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
    """
    exponent = ((bits >> 23) & 0xFF).astype(np.int64)
    mantissa = (bits & 0x7F_FFFF) | (1 << 23)
    right_shift = 16 - (exponent - 127)
    shifted = mantissa >> np.clip(right_shift, 0, 63)
    return np.where((exponent == 0) | (right_shift >= 24), 0, np.where(right_shift <= 0, 0x1FFF, shifted))


def _bf16_round(value: float) -> int:
    """Rounds one float to round-to-nearest-even BF16 bits, matching the table the SFU is programmed from.

    Args:
        value: The value to round.

    Returns:
        The sixteen BF16 bits.
    """
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
