#!/usr/bin/env python3
"""Provides GPT-2 small's inference on the NPU, which includes the device tensors, the launches of every compiled
image, and the forward pass that chains them.

Running this module generates a continuation of a prompt on a board.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import pathlib
import sys
import time

import numpy as np
from numpy.typing import NDArray

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The board package's directory."""

sys.path.insert(0, str(HERE))

from bf16 import (  # noqa: E402
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_DEPTH,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _ELEMENT_BYTES,
    _MODEL_WIDTH,
    _NUM_LANES,
    _ROW_BYTES,
    _TILE,
    _VREG_ELEMENTS,
    _ZERO_TILE_BYTES,
    _ZERO_TILE_WORDS,
    _on_bf16_grid,
    as_bf16_words,
    from_bf16_words,
    pack_activations,
    decode_mask_tiles,
    pack_attention_constants,
    pack_bias,
    pack_weights,
)
from tokenizer import Tokenizer  # noqa: E402
from runtime import (  # noqa: E402
    MemcpyKind,
    minitpuFree,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
)


_TENSOR_POOL: dict[tuple[int, int], list[int]] = {}
"""Freed tensors waiting to be handed out again, keyed by shape. A launch reads a drain buffer's clearing tile
and never writes it, so a reused buffer needs no re-zeroing, which is the copy the pool saves."""


_RESIDENT_IMAGES: dict[bytes, int] = {}
"""What each kernel base currently holds, so relaunching an unchanged image copies nothing. Sound only while
every launch in the process stages through `_stage_image`, which a program mixing in a runtime-staged kernel
would not, and the cache would then belong in the runtime instead."""


_IMAGE_DIRECTORY: pathlib.Path = HERE / "images"
"""Directory holding the compiled images and the manifest describing them, which a host writes through
`tools/board/build_gpt2_images.py`."""


UIO_NAME: str = "minitpu"
"""UIO node every image launches on."""


_COMPILED_IMAGES: dict[tuple, tuple[bytes, dict]] = {}
"""Compiled images and their manifest entries, keyed by kind and shape, each read from disk once."""


_ATTENTION_CONSTANTS: dict[str, int] = {}
"""DMEM address of each attention image's constants block, keyed by its file, each uploaded once."""


_MAX_CONTEXT: int = 1024
"""Tokens GPT-2 small attends over at most, being the positions its position embedding covers."""


_END_OF_TEXT: int = 50256
"""Token a prompt is padded with, being GPT-2's end-of-text token."""


_NPU_CLOCK: list[float] = [0.0]
"""Seconds spent inside launches since the process started, from issue until the NPU retires each image."""

_DECODE_CONSTANTS: dict[int, int] = {}
"""DMEM address of each decode image's constants block, keyed by the cache length it walks, rewritten each step."""


class ResidentWeights:
    """A packed weight matrix that lives in DMEM across launches.

    Notes:
        Re-deriving the device layout on every call costs 72.7 ms of a 168 ms launch, and the weights do not
        change between tokens. Device allocations outlive a launch already, `minitpuHostReset` resetting only
        the host udmabuf, so residency needed somewhere to keep the address and nothing more.

    Attributes:
        address: Device address the packed matrix starts at, or None once freed.
        depth: Contraction the matrix walks.
        columns: Columns the matrix covers.
        nbytes: Bytes the packed matrix occupies.
    """

    def __init__(self, address: int, depth: int, columns: int, nbytes: int) -> None:
        self.address = address
        self.depth = depth
        self.columns = columns
        self.nbytes = nbytes

    @property
    def shape(self) -> tuple[int, int]:
        return (self.depth, self.columns)

    def free(self) -> None:
        """Releases the DMEM this holds."""
        if self.address is not None:
            minitpuFree(self.address)
            self.address = None


class DeviceTensor:
    """Packed activations living in DMEM, in the layout a fused GEMM already drains.

    Notes:
        A fused launch writes [column_tiles, BLOCK_TOKENS, TILE], which is the layout `pack_activations`
        produces, so a product is already the next launch's packed left operand with nothing to repack. The
        layout is block major rather than one array with a token axis, which is what leaves a block's operand
        and its output each a plain offset from the base. Every block is headed by its own zeroed clearing tile,
        a launch loading that tile from the drain pointer it is given.

    Attributes:
        base: Device address the tensor starts at.
        tokens: Token rows the tensor holds, a whole number of blocks.
        features: Features each token row carries.
    """

    def __init__(self, base: int, tokens: int, features: int) -> None:
        self.base = base
        self.tokens = tokens
        self.features = features

    @property
    def blocks(self) -> int:
        return self.tokens // _BLOCK_TOKENS

    @property
    def block_bytes(self) -> int:
        """One block's clearing tile plus its packed rows."""
        return _ZERO_TILE_BYTES + _BLOCK_TOKENS * self.features * _ELEMENT_BYTES

    def d_drain(self, block: int) -> int:
        """The address a launch binds as its drain argument for one block."""
        return self.base + block * self.block_bytes

    def operand(self, block: int) -> int:
        """The address a launch binds as its activation argument for one block."""
        return self.d_drain(block) + _ZERO_TILE_BYTES

    @property
    def address(self) -> int:
        return self.operand(0)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tokens, self.features)

    @property
    def nbytes(self) -> int:
        return self.blocks * self.block_bytes

    def free(self) -> None:
        """Returns this tensor to the pool for reuse.

        Notes:
            Not a `minitpuFree`: a clearing tile is read by a launch and never written by one, so a buffer
            of the same shape is handed straight back out with its tile still zeroed, which saves a host
            copy per allocation.
        """
        if self.base is not None:
            _TENSOR_POOL.setdefault((self.tokens, self.features), []).append(self.base)
            self.base = None


class DeviceTensorSlice:
    """A contiguous feature range of a packed tensor, addressed in ITS blocks.

    Notes:
        A block is `[features / TILE, BLOCK_TOKENS, TILE]`, so a range of feature tiles is a contiguous run
        at a plain offset and splitting a fused QKV projection into thirds copies nothing. This is a view
        rather than a `DeviceTensor` because a tensor derives its block stride from its own feature count,
        while a slice must keep the parent's.

    Attributes:
        tokens: Token rows the slice covers, being the parent's.
        features: Features the slice spans.
    """

    def __init__(self, d_parent: DeviceTensor, first_tile: int, features: int) -> None:
        self._d_parent = d_parent
        self._offset = first_tile * _BLOCK_TOKENS * _TILE * _ELEMENT_BYTES
        self.tokens = d_parent.tokens
        self.features = features

    @property
    def blocks(self) -> int:
        """Returns the token blocks the slice covers, being its parent's."""
        return self._d_parent.blocks

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tokens, self.features)

    def operand(self, block: int) -> int:
        """The address a launch binds as its activation argument for one block."""
        return self._d_parent.operand(block) + self._offset


class KVCache:
    """One layer's keys and values for every token attended so far, resident in the layout decode attention reads.

    Notes:
        The cache is a [capacity, 2 * model] tensor holding each token's keys and then its values, in the block
        layout the QKV projection drains, so filling it is a copy and never a repack.

    Args:
        capacity: Tokens the cache can hold, a whole number of token blocks.
        model: The model width, which each token's keys and values each span.

    Attributes:
        tensor: The resident keys and values.
        length: Tokens the cache holds.
    """

    def __init__(self, capacity: int, model: int = _MODEL_WIDTH) -> None:
        self.tensor = _allocate_tensor(capacity, 2 * model)
        self.length = 0

    def extend(self, d_fused: DeviceTensor, *, tokens: int | None = None) -> None:
        """Copies a prefill's keys and values into the cache, after the tokens it already holds.

        Args:
            d_fused: [tokens, 3 * model] fused QKV projection, resident.
            tokens: Tokens of it that are real, the rest padding its last block, or None for every one.

        Raises:
            ValueError: If the cache holds part of a token block, or has no room for the prefill.
        """
        tokens = d_fused.tokens if tokens is None else tokens
        if self.length % _BLOCK_TOKENS or self.length + d_fused.tokens > self.tensor.tokens:
            message = (
                f"Unable to extend a cache holding {self.length} of {self.tensor.tokens} tokens by {d_fused.tokens}. "
                f"A prefill fills whole token blocks, so the cache must hold whole blocks and have room for them."
            )
            raise ValueError(message)
        rows = self.tensor.features // _TILE * _BLOCK_TOKENS
        first = self.length // _BLOCK_TOKENS
        for block in range(d_fused.blocks):
            _launch_copy(rows=rows, stride=_ROW_BYTES, source=d_fused.operand(block) + self._keys_offset(d_fused),
                         destination=self.tensor.operand(first + block))
        self.length += tokens

    def append(self, d_fused: DeviceTensor) -> None:
        """Copies a decode step's new token, the first row of its fused QKV projection, into the next slot.

        Args:
            d_fused: [tokens, 3 * model] fused QKV projection of the decode step, resident.

        Raises:
            ValueError: If the cache is full.
        """
        if self.length >= self.tensor.tokens:
            message = f"Unable to append to a cache holding all {self.tensor.tokens} tokens it has room for."
            raise ValueError(message)
        block, row = divmod(self.length, _BLOCK_TOKENS)
        _launch_copy(rows=self.tensor.features // _TILE, stride=_BLOCK_TOKENS * _ROW_BYTES,
                     source=d_fused.operand(0) + self._keys_offset(d_fused),
                     destination=self.tensor.operand(block) + row * _ROW_BYTES)
        self.length += 1

    def free(self) -> None:
        """Returns the cache's tensor to the pool."""
        self.tensor.free()

    @staticmethod
    def _keys_offset(d_fused: DeviceTensor) -> int:
        """Returns the bytes a token block's keys lie past its queries, being one model width of tiles."""
        return d_fused.features // 3 // _TILE * _BLOCK_TOKENS * _ROW_BYTES


def npu_seconds() -> float:
    """Returns the seconds spent inside launches so far, which excludes every upload, download and host step."""
    return _NPU_CLOCK[0]


def halt_image() -> bytes:
    """Returns the compiled image that halts at once, being the shortest program a launch can run."""
    return _compiled_image(kind="halt")[0]


def load_gpt2_weights(path: pathlib.Path | None = None) -> tuple[list[dict], dict]:
    """Reads the weights `tools/board/export_gpt2_weights.py` wrote, as float32 arrays.

    Args:
        path: The exported weights file, or None for the one in this package's weights directory.

    Returns:
        One dict per layer, keyed as `transformer_block` reads its parameters, and the shared weights, keyed
        `wte`, `wpe`, `ln_f.g` and `ln_f.b`.

    Raises:
        FileNotFoundError: If the weights have not been exported.
    """
    path = HERE / "weights" / "gpt2_weights.npz" if path is None else path
    if not path.exists():
        message = (
            f"Unable to read GPT-2's weights from {path}. Run tools/board/export_gpt2_weights.py on a host, which writes "
            f"them there, and deploy the package again."
        )
        raise FileNotFoundError(message)
    exported = np.load(path)
    weights = {name: from_bf16_words(words=exported[name]) for name in exported.files}
    layers: list[dict] = []
    while f"h.{len(layers)}.ln_1.g" in weights:
        prefix = f"h.{len(layers)}."
        layers.append({name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)})
    shared = {name: value for name, value in weights.items() if not name.startswith("h.")}
    return layers, shared


def upload_weights(h_weights: NDArray[np.float32]) -> ResidentWeights:
    """Packs a weight matrix and leaves it in DMEM for repeated launches.

    Args:
        h_weights: [contraction, columns] host operand, rows being the contraction.

    Returns:
        A handle `full_gemm` accepts in place of the array.
    """
    words = pack_weights(weights=_on_bf16_grid(values=h_weights.astype(np.float32)))
    minitpuSetDevice(UIO_NAME)
    host = minitpuHostAlloc(words.nbytes)
    host.view.view("<u2")[: words.size] = words
    # One block of slack for the kernel's last prefetch, which reads one block past
    # the final block.
    base = minitpuMalloc(words.nbytes + _BLOCK_WEIGHT_BYTES)
    minitpuMemcpy(base, host, words.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    return ResidentWeights(address=base, depth=h_weights.shape[0],
                           columns=h_weights.shape[1], nbytes=words.nbytes)


def upload_bias(h_bias: NDArray[np.float32]) -> ResidentWeights:
    """Uploads one feature row, replicated so a stride-one VMEM read broadcasts it.

    A VREG holds `_NUM_SUBLANES` token rows of one feature tile, and every one of them wants the same tile of the
    bias, so repeating each tile that many times in DMEM makes that a plain contiguous read.
    """
    if h_bias.ndim != 1 or h_bias.size % _NUM_LANES:
        raise ValueError(
            f"Unable to upload bias shape {h_bias.shape}. It must be one row of "
            f"whole {_NUM_LANES}-lane tiles."
        )
    words = pack_bias(_on_bf16_grid(values=h_bias.astype(np.float32)))
    minitpuSetDevice(UIO_NAME)
    host = minitpuHostAlloc(words.nbytes)
    host.view.view("<u2")[:words.size] = words
    address = minitpuMalloc(words.nbytes)
    minitpuMemcpy(address, host, words.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    return ResidentWeights(address=address, depth=1, columns=h_bias.size,
                           nbytes=words.nbytes)


def upload_activations(h_activations: NDArray[np.float32]) -> DeviceTensor:
    """Packs host activations and leaves them in DMEM.

    Args:
        h_activations: [tokens, features] host array, tokens a whole number of the block.

    Returns:
        A tensor a device launch can consume without any further copy.
    """
    tokens, features = h_activations.shape
    h_grid = _on_bf16_grid(values=h_activations.astype(np.float32))
    d_tensor = _allocate_tensor(tokens, features)
    for block in range(d_tensor.blocks):
        start = block * _BLOCK_TOKENS
        words = pack_activations(activations=h_grid[start:start + _BLOCK_TOKENS])
        h_staging = minitpuHostAlloc(words.nbytes)
        h_staging.view.view("<u2")[: words.size] = words
        minitpuMemcpy(d_tensor.operand(block), h_staging, words.nbytes, MemcpyKind.HostToDevice)
    return d_tensor


def download_activations(d_tensor: DeviceTensor) -> NDArray[np.float32]:
    """Brings a device tensor back as a row-major float32 array.

    Only the boundaries need this: the embedding's output goes in, the language
    head's input comes out, and nothing in between has to make the trip.

    Args:
        d_tensor: The device tensor to read.

    Returns:
        Its [tokens, features] contents, on the host.
    """
    per_block = _BLOCK_TOKENS * d_tensor.features
    h_out = np.empty((d_tensor.tokens, d_tensor.features), dtype=np.float32)
    for block in range(d_tensor.blocks):
        h_staging = minitpuHostAlloc(per_block * _ELEMENT_BYTES)
        minitpuMemcpy(h_staging, d_tensor.operand(block), per_block * _ELEMENT_BYTES,
                      MemcpyKind.DeviceToHost)
        values = from_bf16_words(words=h_staging.view.view("<u2")[:per_block])
        tiles = values.reshape(d_tensor.features // _TILE, _BLOCK_TOKENS, _TILE)
        h_out[block * _BLOCK_TOKENS:(block + 1) * _BLOCK_TOKENS] = (
            tiles.transpose(1, 0, 2).reshape(_BLOCK_TOKENS, d_tensor.features))
    return h_out


# Host staging takes a fresh allocation per call deliberately. The allocator is a bump allocator with no
# per-region free, so a process sweeping many shapes exhausts its 16 MiB udmabuf and a benchmark takes one
# shape per process. Caching a region per role does not survive `minitpuHostReset`, which the drivers call
# and which drops the udmabuf out from under every region carved from it. The fix belongs in the allocator.
def upload_layernorm_constants(h_gamma: NDArray[np.float32],
                               h_beta: NDArray[np.float32]) -> int:
    """Uploads one LayerNorm's constants block and returns its DMEM address.

    Notes:
        The block is gamma and beta followed by every number the compiled image states, each broadcast across one
        VREG, in the order its manifest entry lists them.

    Args:
        h_gamma: Per-feature scale, on the host.
        h_beta: Per-feature shift, on the host.

    Returns:
        The device address the kernel binds as its constants argument.

    Raises:
        ValueError: If no compiled LayerNorm covers the width, or its image reads other rows than gamma and beta.
    """
    _, entry = _compiled_image(kind="layernorm", features=h_gamma.size)
    if entry["rows"] != ["gamma_pointer", "beta_pointer"]:
        message = (
            f"Unable to pack the constants of a LayerNorm image reading the rows {entry['rows']}. These drivers "
            f"pack gamma and then beta, so the image must read exactly those two."
        )
        raise ValueError(message)
    broadcast = [np.full(_VREG_ELEMENTS, scalar, dtype=np.float32) for scalar in entry["scalars"]]
    h_constants = as_bf16_words(values=np.concatenate([h_gamma, h_beta, *broadcast]))
    h_staging = minitpuHostAlloc(h_constants.nbytes)
    h_staging.view.view("<u2")[: h_constants.size] = h_constants
    d_address = minitpuMalloc(h_constants.nbytes)
    minitpuMemcpy(d_address, h_staging, h_constants.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    return d_address


def layernorm_resident(d_rows: DeviceTensor, d_constants: int, *,
                       d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Normalises a resident tensor through the compiled LayerNorm image, leaving it resident.

    Notes:
        The image reads and writes in the layout a GEMM operand uses, so a normalised tensor feeds the next
        projection where it lies rather than returning to PS DDR to be re-packed.

    Args:
        d_rows: [tokens, model] resident rows.
        d_constants: DMEM address returned by `upload_layernorm_constants`.
        d_out: Where to write. Allocated when omitted.

    Returns:
        The normalised rows, resident.

    Raises:
        ValueError: If no compiled LayerNorm covers the rows' width.
    """
    image_bytes, _ = _compiled_image(kind="layernorm", features=d_rows.features)
    if d_out is None:
        d_out = _allocate_tensor(d_rows.tokens, d_rows.features)

    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    for block in range(d_rows.blocks):
        _launch(
            d_image, len(image_bytes),
            args=[d_rows.operand(block), d_constants, d_out.operand(block)])
    return d_out


def add_resident(d_left: DeviceTensor, d_right: DeviceTensor, *,
                 d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Adds two DMEM tensors elementwise, leaving the sum in DMEM.

    Notes:
        This is what a projection's bias and a block's residual become once they stop being numpy, each of
        which otherwise costs two host copies per add for one `vadd` the machine has always had.

    Args:
        d_left: A resident tensor.
        d_right: A resident tensor of the same shape.
        d_out: Where to write. Allocated when omitted.

    Returns:
        The sum, resident.

    Raises:
        ValueError: If the operands disagree in shape, or no compiled image covers their width.
    """
    if d_left.shape != d_right.shape:
        message = (
            f"Unable to add a {d_left.shape} tensor to a {d_right.shape} one. "
            f"Elementwise addition needs the same shape."
        )
        raise ValueError(message)
    image_bytes, _ = _compiled_image(kind="add", features=d_left.features)
    if d_out is None:
        d_out = _allocate_tensor(d_left.tokens, d_left.features)

    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    for block in range(d_left.blocks):
        _launch(
            d_image, len(image_bytes),
            args=[d_left.operand(block), d_right.operand(block), d_out.operand(block)])
    return d_out


def add_bias_resident(d_left: DeviceTensor, d_bias: ResidentWeights, *,
                      d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Adds one resident feature row to every token row.

    Notes:
        The bias is packed with each feature tile repeated across the four sublane rows, so the kernel stages
        it once and one contiguous read broadcasts it across tokens, rather than materialising a
        `[tokens, features]` tensor per layer and shape.

    Raises:
        ValueError: If the bias is not one row of the tensor's width, or no compiled image covers that width.
    """
    if d_bias.depth != 1 or d_bias.columns != d_left.features:
        raise ValueError(
            f"Unable to broadcast bias {d_bias.shape} over {d_left.shape}. The "
            "bias must be one row with the tensor's feature width."
        )
    image_bytes, _ = _compiled_image(kind="add_bias", features=d_left.features)
    if d_out is None:
        d_out = _allocate_tensor(d_left.tokens, d_left.features)

    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    for block in range(d_left.blocks):
        _launch(
            d_image, len(image_bytes),
            args=[d_left.operand(block), d_bias.address, d_out.operand(block)])
    return d_out


def gemm_resident(d_activations: DeviceTensor, d_weights: ResidentWeights, *,
                  activate: bool = False, d_bias: ResidentWeights | None = None,
                  d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Multiplies two DMEM operands and leaves the product in DMEM.

    Notes:
        Nothing crosses the host boundary: both operands are already packed and the product is written in the
        layout the next launch's activation argument expects, leaving the host only the launch to issue.

    Notes:
        Always fuses the contraction's groups, a per-group drain leaving partials the host would have to add,
        which costs about 1.8x the relative error of the float32 host sum. One launch covers a block of
        `_BLOCK_TOKENS` rows, the activation pointer advancing by a block of packed rows and the drain pointer
        by a block of output rows.

    Args:
        d_activations: [tokens, contraction] left operand, resident.
        d_weights: [contraction, columns] right operand, resident.
        activate: Whether each staged tile is GELU-activated entering the array.
        d_bias: One resident row added to every product row, as `upload_bias` leaves it, or None for none. An image
            set that fuses the bias into the product adds it there, and any other adds it in a launch of its own.
        d_out: Where to write. Allocated when omitted.

    Returns:
        The product, resident.

    Raises:
        ValueError: If the operands disagree, if a dimension is not a whole number of blocks, or if no compiled image
            covers the shape.
    """
    tokens, depth = d_activations.shape
    columns = d_weights.columns
    if depth != d_weights.depth:
        message = (
            f"Unable to multiply a [{tokens},{depth}] tensor by a [{d_weights.depth},{columns}] "
            f"one. Their inner dimensions must agree."
        )
        raise ValueError(message)
    for name, value, block in (("tokens", tokens, _BLOCK_TOKENS), ("columns", columns, _TILE)):
        if value % block:
            message = (
                f"Unable to multiply tensors with {name} of {value}. It must be a whole number "
                f"of {block}, which is what one launch covers."
            )
            raise ValueError(message)

    fused = d_bias is not None and _has_image(kind="gemm_bias", depth=depth, columns=columns, activate=activate)
    if d_bias is not None and not fused:
        d_product = gemm_resident(d_activations, d_weights, activate=activate)
        try:
            return add_bias_resident(d_product, d_bias, d_out=d_out)
        finally:
            d_product.free()
    image_bytes = _compiled_gemm(depth=depth, columns=columns, activate=activate, bias=fused)

    if d_out is None:
        d_out = _allocate_tensor(tokens, columns)

    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)

    bias_arguments = [d_bias.address] if fused else []
    for block in range(tokens // _BLOCK_TOKENS):
        _launch(
            d_image, len(image_bytes),
            args=[d_activations.operand(block), d_weights.address, d_out.d_drain(block), *bias_arguments])
    return d_out


def full_gemm(
    activations: NDArray[np.float32], weights: NDArray[np.float32] | ResidentWeights,
    *, activate: bool = False
) -> NDArray[np.float32]:
    """Multiplies two arrays on the device through the compiled image for their shape, one launch per token block.

    Notes:
        The device walks every output column tile and sums the whole contraction in its BF16 accumulators, so
        each launch drains the finished product once per column tile.

        `weights` may be a `ResidentWeights` handle rather than an array, in which case nothing is packed or
        copied, the matrix already being in DMEM. Packing is the single largest cost of a launch, so a caller
        reusing a matrix should hand one over.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction.
        weights: The right operand, whose rows are the contraction and columns the output features.
        activate: Determines whether each staged tile is activated before it enters the array. Set it for a
            feed-forward layer's down projection, whose left operand is the activated intermediate.

    Returns:
        The float32 product, recovered from the BF16 values the kernel returns.

    Raises:
        ValueError: If the operands' inner dimensions disagree, if a dimension is not a whole number of the
            blocks one launch covers, or if no compiled image covers the shape.
    """
    tokens, depth = activations.shape
    resident = weights if isinstance(weights, ResidentWeights) else None
    columns = weights.shape[1]
    if depth != weights.shape[0]:
        message = (
            f"Unable to multiply a [{tokens},{depth}] array by a [{weights.shape[0]},{columns}] array. Their "
            f"inner dimensions must agree."
        )
        raise ValueError(message)
    for name, value, block in (
        ("tokens", tokens, _BLOCK_TOKENS), ("depth", depth, _BLOCK_DEPTH), ("columns", columns, _TILE)
    ):
        if value % block:
            message = (
                f"Unable to multiply arrays with {name} of {value}. It must be a whole number of {block}, which "
                f"is what one launch covers."
            )
            raise ValueError(message)
    column_tiles = columns // _TILE

    image_bytes = _compiled_gemm(depth=depth, columns=columns, activate=activate)
    weight_words = None if resident is not None else pack_weights(weights=weights)
    drain_words = _ZERO_TILE_WORDS + column_tiles * _BLOCK_TOKENS * _TILE
    result = np.empty((tokens, columns), dtype=np.float32)

    minitpuSetDevice(UIO_NAME)
    minitpuHostReset()
    d_owned: list[int] = []

    try:
        d_image = _stage_image(image_bytes)

        if resident is not None:
            # Already in DMEM, and deliberately not appended to `device`: it must
            # outlive this launch, which is the whole point of uploading it once.
            d_weights = resident.address
        else:
            h_weights = minitpuHostAlloc(weight_words.nbytes)
            h_weights.view.view("<u2")[: weight_words.size] = weight_words
            # One block of slack for the kernel's last prefetch, which reads one block past the final block.
            d_weights = minitpuMalloc(weight_words.nbytes + _BLOCK_WEIGHT_BYTES)
            d_owned.append(d_weights)
            minitpuMemcpy(d_weights, h_weights, weight_words.nbytes, MemcpyKind.HostToDevice)

        # The clearing tile heads the drain buffer, so one argument reaches both and the kernel steps past the
        # tile to find where the product lands.
        h_drain = minitpuHostAlloc(drain_words * _ELEMENT_BYTES)
        h_drain.view.view("<u2")[:drain_words] = 0
        d_drain = minitpuMalloc(drain_words * _ELEMENT_BYTES)
        d_owned.append(d_drain)
        minitpuMemcpy(d_drain, h_drain, drain_words * _ELEMENT_BYTES, MemcpyKind.HostToDevice)

        h_activations = minitpuHostAlloc(_BLOCK_TOKENS * depth * _ELEMENT_BYTES)
        d_activations = minitpuMalloc(_BLOCK_TOKENS * depth * _ELEMENT_BYTES + _BLOCK_ACTIVATION_BYTES)
        d_owned.append(d_activations)

        for token in range(0, tokens, _BLOCK_TOKENS):
            block_words = pack_activations(activations=activations[token : token + _BLOCK_TOKENS])
            h_activations.view.view("<u2")[: block_words.size] = block_words
            minitpuMemcpy(d_activations, h_activations, block_words.nbytes, MemcpyKind.HostToDevice)

            _launch(d_image, len(image_bytes),
                                args=[d_activations, d_weights, d_drain])
            minitpuMemcpy(h_drain, d_drain, drain_words * _ELEMENT_BYTES, MemcpyKind.DeviceToHost)

            tiles = from_bf16_words(
                words=h_drain.view.view("<u2")[_ZERO_TILE_WORDS:drain_words]).reshape(
                column_tiles, _BLOCK_TOKENS, _TILE)
            result[token : token + _BLOCK_TOKENS] = tiles.transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()
    return result


def transformer_block(hidden: NDArray[np.float32], weights: dict, *, heads: int = 12,
                      cache: KVCache | None = None, valid: int | None = None) -> NDArray[np.float32]:
    """Runs one GPT-2 transformer block over a sequence on the device, keeping every intermediate resident in DMEM.

    Args:
        hidden: [tokens, model] block input, a sequence length a compiled attention image covers.
        weights: The block's parameters, keyed as GPT-2 names them: ``ln_1.g``, ``ln_1.b``,
            ``attn.qkv.w``, ``attn.qkv.b``, ``attn.proj.w``, ``attn.proj.b``, ``ln_2.g``,
            ``ln_2.b``, ``mlp.fc.w``, ``mlp.fc.b``, ``mlp.proj.w``, ``mlp.proj.b``.
        heads: Attention heads the model width splits into.
        cache: The layer's KV cache, which the block's keys and values are copied into, or None for no cache.
        valid: Rows of the input that are real tokens, the rest padding, which is the length the cache records.
            None counts every row as real.

    Returns:
        The [tokens, model] block output.
    """
    def attend(d_fused: DeviceTensor) -> DeviceTensorSlice:
        """Fills the cache from the sequence's keys and values, then attends the sequence causally."""
        if cache is not None:
            cache.extend(d_fused, tokens=valid)
        return attention(d_fused, heads=heads)

    return _run_layer(hidden, weights, heads=heads, attend=attend)


def decode_block(hidden: NDArray[np.float32], weights: dict, cache: KVCache, *,
                 heads: int = 12) -> NDArray[np.float32]:
    """Runs one decode step's new token through one GPT-2 block on the device, attending over the layer's cache.

    Args:
        hidden: [model] the new token's block input.
        weights: The block's parameters, keyed as `transformer_block` reads them.
        cache: The layer's KV cache, which the new token's keys and values are appended to.
        heads: Attention heads the model width splits into.

    Returns:
        The [model] block output for the new token.
    """
    # Pads the token to the token block every launch covers, its rows independent outside attention.
    padded = np.zeros((_BLOCK_TOKENS, hidden.shape[-1]), dtype=np.float32)
    padded[0] = hidden

    def attend(d_fused: DeviceTensor) -> DeviceTensorSlice:
        """Appends the new token's keys and values to the cache, then attends it over every cached token."""
        cache.append(d_fused)
        return decode_attention(d_fused, cache, heads=heads)

    return _run_layer(padded, weights, heads=heads, attend=attend)[0]


def embed(token_ids: NDArray[np.int64], shared: dict, *, start: int = 0) -> NDArray[np.float32]:
    """Returns the input the first layer reads, each token's embedding plus its position's.

    Args:
        token_ids: The tokens, in order.
        shared: The shared weights `load_gpt2_weights` returns, which hold `wte` and `wpe`.
        start: Position of the first token, which a decode step sets past the tokens already attended.

    Returns:
        The [tokens, model] embeddings.
    """
    positions = start + np.arange(len(token_ids))
    return (shared["wte"][np.asarray(token_ids)] + shared["wpe"][positions]).astype(np.float32)


def prefill(token_ids: NDArray[np.int64], layers: list[dict], shared: dict, *, heads: int = 12,
            capacity: int = _MAX_CONTEXT,
            observe: Callable[[int, NDArray[np.float32], NDArray[np.float32]], None] | None = None,
            ) -> tuple[NDArray[np.float32], list[KVCache]]:
    """Runs a prompt through every layer on the device, filling one KV cache per layer.

    Notes:
        The prompt is padded at its end to the shortest sequence length a compiled attention image covers. Attention
        is causal, so no real token reads a padded one, and each cache records only the real tokens.

    Args:
        token_ids: The prompt's tokens, in order.
        layers: Each layer's weights, as `load_gpt2_weights` returns them.
        shared: The shared weights, as `load_gpt2_weights` returns them.
        heads: Attention heads.
        capacity: Tokens each cache holds room for, which bounds the prompt plus every decoded token.
        observe: Called with each layer's index, its padded input and its padded output, or None.

    Returns:
        The last layer's output for the real tokens, and each layer's cache, which the caller frees.

    Raises:
        ValueError: If no compiled attention image covers the prompt, or the caches cannot hold it.
    """
    tokens = len(token_ids)
    lengths = sorted(entry["tokens"] for entry in _manifest().values() if entry["kind"] == "attention")
    padded = next((length for length in lengths if length >= tokens), None)
    if padded is None or padded > capacity:
        message = (
            f"Unable to prefill a prompt of {tokens} tokens. A compiled attention image covers "
            f"{', '.join(map(str, lengths))} tokens and the caches hold {capacity}, so add a longer length to "
            f"tools/board/build_gpt2_images.py or shorten the prompt."
        )
        raise ValueError(message)
    padding = np.full(padded - tokens, _END_OF_TEXT, dtype=np.int64)
    hidden = embed(np.concatenate([np.asarray(token_ids, dtype=np.int64), padding]), shared)
    caches = [KVCache(capacity=capacity) for _ in layers]
    try:
        for index, (weights, cache) in enumerate(zip(layers, caches, strict=True)):
            layer_input = hidden
            hidden = transformer_block(hidden, weights, heads=heads, cache=cache, valid=tokens)
            if observe is not None:
                observe(index, layer_input, hidden)
    except BaseException:
        for cache in caches:
            cache.free()
        raise
    return hidden[:tokens], caches


def decode_step(token_id: int, layers: list[dict], shared: dict, caches: list[KVCache], *, heads: int = 12,
                observe: Callable[[int, NDArray[np.float32], NDArray[np.float32]], None] | None = None,
                ) -> NDArray[np.float32]:
    """Runs one new token through every layer on the device, appending its keys and values to each layer's cache.

    Args:
        token_id: The new token.
        layers: Each layer's weights, as `load_gpt2_weights` returns them.
        shared: The shared weights, as `load_gpt2_weights` returns them.
        caches: Each layer's KV cache, holding every earlier token, whose length is the new token's position.
        heads: Attention heads.
        observe: Called with each layer's index, its input for the new token and its output, or None.

    Returns:
        The [model] last layer's output for the new token.
    """
    hidden = embed(np.asarray([token_id]), shared, start=caches[0].length)[0]
    for index, (weights, cache) in enumerate(zip(layers, caches, strict=True)):
        layer_input = hidden
        hidden = decode_block(hidden, weights, cache, heads=heads)
        if observe is not None:
            observe(index, layer_input, hidden)
    return hidden


def final_layernorm(hidden: NDArray[np.float32], shared: dict) -> NDArray[np.float32]:
    """Applies GPT-2's final LayerNorm to token rows on the device, through the compiled LayerNorm image.

    Args:
        hidden: [tokens, model] the last layer's output.
        shared: The shared weights, as `load_gpt2_weights` returns them, which hold `ln_f.g` and `ln_f.b`.

    Returns:
        The [tokens, model] normalised rows.
    """
    tokens = hidden.shape[0]
    padded = np.zeros((-(-tokens // _BLOCK_TOKENS) * _BLOCK_TOKENS, hidden.shape[1]), dtype=np.float32)
    padded[:tokens] = hidden
    d_rows, d_constants = upload_activations(padded), upload_layernorm_constants(shared["ln_f.g"], shared["ln_f.b"])
    try:
        d_normed = layernorm_resident(d_rows, d_constants)
        try:
            return download_activations(d_normed)[:tokens]
        finally:
            d_normed.free()
    finally:
        d_rows.free()
        minitpuFree(d_constants)


def lm_head(normalised: NDArray[np.float32], shared: dict) -> NDArray[np.float32]:
    """Scores every vocabulary token against one normalised row through the tied embedding.

    Args:
        normalised: [model] the final LayerNorm's output for the token being predicted from.
        shared: The shared weights, as `load_gpt2_weights` returns them, which hold `wte`.

    Returns:
        The [vocabulary] logits.
    """
    return normalised @ shared["wte"].T


def next_token_logits(hidden: NDArray[np.float32], shared: dict) -> NDArray[np.float32]:
    """Scores every vocabulary token after one token, the final LayerNorm on the device and the head on the host.

    Args:
        hidden: [model] the last layer's output for the token being predicted from.
        shared: The shared weights, as `load_gpt2_weights` returns them.

    Returns:
        The [vocabulary] logits.
    """
    return lm_head(final_layernorm(hidden[None, :], shared)[0], shared)


def generate(token_ids: NDArray[np.int64], layers: list[dict], shared: dict, *, new_tokens: int, heads: int = 12,
             capacity: int = _MAX_CONTEXT) -> list[int]:
    """Greedily generates a prompt's continuation, prefilling the prompt and decoding one token per step.

    Args:
        token_ids: The prompt's tokens, in order.
        layers: Each layer's weights, as `load_gpt2_weights` returns them.
        shared: The shared weights, as `load_gpt2_weights` returns them.
        new_tokens: Tokens to generate at most, generation stopping early at GPT-2's end-of-text token.
        heads: Attention heads.
        capacity: Tokens each layer's cache holds room for.

    Returns:
        The generated tokens, in order.

    Raises:
        ValueError: If the prompt and its continuation outgrow the caches.
    """
    if len(token_ids) + new_tokens > capacity:
        message = (
            f"Unable to generate {new_tokens} tokens after a {len(token_ids)}-token prompt. The caches hold "
            f"{capacity} tokens, so ask for fewer."
        )
        raise ValueError(message)
    hidden, caches = prefill(token_ids, layers, shared, heads=heads, capacity=capacity)
    generated: list[int] = []
    try:
        logits = next_token_logits(hidden[-1], shared)
        while len(generated) < new_tokens:
            token = int(np.argmax(logits))
            generated.append(token)
            if token == _END_OF_TEXT or len(generated) == new_tokens:
                break
            logits = next_token_logits(decode_step(token, layers, shared, caches, heads=heads), shared)
    finally:
        for cache in caches:
            cache.free()
    return generated


def attention(d_fused: DeviceTensor, *, heads: int = 12) -> DeviceTensorSlice:
    """Attends every head of a resident fused QKV tensor causally, in one launch of the compiled image.

    Notes:
        The image reads each head's queries, keys and values where the QKV projection drained them and writes the
        output over the queries, so the attention output is the fused tensor's first third.

    Args:
        d_fused: [tokens, 3 * model] fused QKV projection, resident.
        heads: Attention heads.

    Returns:
        The [tokens, model] attention output, a view of the fused tensor.

    Raises:
        ValueError: If no compiled attention image covers the sequence length and head count.
    """
    image_bytes, entry = _compiled_image(kind="attention", tokens=d_fused.tokens, heads=heads)
    d_constants = _attention_constants(entry)
    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    _launch(d_image, len(image_bytes), args=[d_constants, 0, d_fused.operand(0)])
    return DeviceTensorSlice(d_fused, 0, d_fused.features // 3)


def decode_attention(d_fused: DeviceTensor, cache: KVCache, *, heads: int = 12) -> DeviceTensorSlice:
    """Attends every head of a decode step's new token over the layer's KV cache, in one launch.

    Notes:
        The step uses the shortest decode image whose length holds the cache, and masks the keys past the cache's
        length with tiles written here, so one image serves every length up to its own.

    Args:
        d_fused: [tokens, 3 * model] fused QKV projection of the step, the new token in its first row, resident.
        cache: The layer's KV cache, already holding the new token's keys and values.
        heads: Attention heads.

    Returns:
        The [tokens, model] attention output, a view of the fused tensor whose first row is the new token's.

    Raises:
        ValueError: If the cache is empty, no decode image holds its length, or that image reads past its capacity.
    """
    built = sorted(entry["keys"] for entry in _manifest().values()
                   if entry["kind"] == "decode_attention" and entry["heads"] == heads)
    keys = next((length for length in built if length >= cache.length), None)
    if cache.length < 1 or keys is None or keys > cache.tensor.tokens:
        message = (
            f"Unable to attend over a cache holding {cache.length} of {cache.tensor.tokens} tokens with decode images "
            f"over {built} keys. A step reads at least one key, through the shortest image holding the cache, and "
            f"that image's keys must fit in the cache's room."
        )
        raise ValueError(message)
    image_bytes, entry = _compiled_image(kind="decode_attention", keys=keys, heads=heads)
    d_constants = _decode_constants(entry, length=cache.length)
    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    _launch(d_image, len(image_bytes), args=[d_constants, cache.tensor.operand(0), d_fused.operand(0)])
    return DeviceTensorSlice(d_fused, 0, d_fused.features // 3)


def _compiled_gemm(*, depth: int, columns: int, activate: bool, bias: bool = False) -> bytes:
    """Returns the compiled image that runs one GEMM shape with the contraction summed on the device.

    Args:
        depth: Contraction depth the product covers.
        columns: Output columns the product covers.
        activate: Determines whether the left operand passes through the GELU before it enters the array.
        bias: Determines whether the image adds the bias row its fourth argument points to.

    Returns:
        The image bytes, padded to the loader's boundary.

    Raises:
        ValueError: If no compiled image covers the shape, or if the image drains other than one partial per
            column tile for a block of token rows.
    """
    image_bytes, entry = _compiled_image(kind="gemm_bias" if bias else "gemm", depth=depth, columns=columns,
                                         activate=activate)
    # Holds the image to the drain these drivers allocate, since an image draining any other layout writes its
    # product into the wrong rows rather than failing.
    expected = _ZERO_TILE_WORDS + columns // _TILE * _BLOCK_TOKENS * _TILE
    if entry["drain_words"] != expected:
        message = (
            f"Unable to launch the [{depth}, {columns}] GEMM image. A product drains {expected} words, one partial "
            f"per column tile, but the image drains {entry['drain_words']}."
        )
        raise ValueError(message)
    return image_bytes


def _compiled_image(*, kind: str, **shape: int | bool) -> tuple[bytes, dict]:
    """Returns the compiled image of one kind and shape, and the manifest entry describing it.

    Args:
        kind: The kind of launch the image runs, as the manifest names it.
        shape: The fields the image's entry must carry, such as its depth, columns or features.

    Returns:
        The image bytes padded to the loader's boundary, and its manifest entry.

    Raises:
        ValueError: If no compiled image carries that kind and shape.
    """
    key = (kind, *sorted(shape.items()))
    cached = _COMPILED_IMAGES.get(key)
    if cached is not None:
        return cached

    manifest = _manifest()
    found = next((name for name, entry in manifest.items()
                  if entry["kind"] == kind and all(entry.get(field) == value for field, value in shape.items())), None)
    if found is None:
        stated = ", ".join(f"{field}={value}" for field, value in shape.items())
        message = (
            f"Unable to launch the {kind} image with {stated}. No image in {_IMAGE_DIRECTORY} covers that shape, so "
            f"add it to tools/board/build_gpt2_images.py and rerun that tool on a host."
        )
        raise ValueError(message)
    cached = ((_IMAGE_DIRECTORY / found).read_bytes(), manifest[found])
    _COMPILED_IMAGES[key] = cached
    return cached


def _has_image(*, kind: str, **shape: int | bool) -> bool:
    """Determines whether the image set carries an image of one kind and shape."""
    return any(entry["kind"] == kind and all(entry.get(field) == value for field, value in shape.items())
               for entry in _manifest().values())


def _attention_constants(entry: dict) -> int:
    """Uploads an attention image's constants block once and returns its DMEM address.

    Args:
        entry: The image's manifest entry, which names the file the block was compiled into.

    Returns:
        The device address the image binds as its constants argument.
    """
    d_cached = _ATTENTION_CONSTANTS.get(entry["constants"])
    if d_cached is not None:
        return d_cached
    h_words = np.frombuffer((_IMAGE_DIRECTORY / entry["constants"]).read_bytes(), dtype="<u2")
    h_staging = minitpuHostAlloc(h_words.nbytes)
    h_staging.view.view("<u2")[: h_words.size] = h_words
    d_address = minitpuMalloc(h_words.nbytes)
    minitpuMemcpy(d_address, h_staging, h_words.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    _ATTENTION_CONSTANTS[entry["constants"]] = d_address
    return d_address


def _run_layer(hidden: NDArray[np.float32], weights: dict, *, heads: int,
               attend: Callable[[DeviceTensor], DeviceTensorSlice]) -> NDArray[np.float32]:
    """Runs one GPT-2 block on the device around the attention it is given, every intermediate resident in DMEM.

    Notes:
        Both LayerNorms, all four projections with their biases, both residual adds and the attention run as
        launches that read the previous launch's output where it lies, so the block crosses the host boundary once on
        entry and once on exit. Attention's 1/sqrt(head) rides the query columns of the QKV projection, folded
        into its weights and bias before upload.

    Args:
        hidden: [tokens, model] block input, a whole number of token blocks.
        weights: The block's parameters, keyed as `transformer_block` reads them.
        heads: Attention heads the model width splits into.
        attend: Attends the fused QKV projection, returning the attention output resident.

    Returns:
        The [tokens, model] block output.
    """
    model = hidden.shape[1]
    scale = np.ones(3 * model, dtype=np.float32)
    scale[:model] = 1.0 / math.sqrt(model // heads)

    d_owned: list[DeviceTensor | ResidentWeights] = []
    d_constants: list[int] = []

    def kept(d_tensor):
        """Records a device allocation so the block frees it however it exits."""
        d_owned.append(d_tensor)
        return d_tensor

    try:
        d_weights = {name: kept(upload_weights(weights[name] * (scale if name == "attn.qkv.w" else 1.0)))
                     for name in ("attn.qkv.w", "attn.proj.w", "mlp.fc.w", "mlp.proj.w")}
        d_biases = {name: kept(upload_bias(weights[name] * (scale if name == "attn.qkv.b" else 1.0)))
                    for name in ("attn.qkv.b", "attn.proj.b", "mlp.fc.b", "mlp.proj.b")}
        for gamma, beta in (("ln_1.g", "ln_1.b"), ("ln_2.g", "ln_2.b")):
            d_constants.append(upload_layernorm_constants(weights[gamma], weights[beta]))

        d_hidden = kept(upload_activations(hidden))
        d_normed = kept(layernorm_resident(d_hidden, d_constants[0]))
        d_fused = kept(gemm_resident(d_normed, d_weights["attn.qkv.w"], d_bias=d_biases["attn.qkv.b"]))
        d_attended = attend(d_fused)
        d_projected = kept(gemm_resident(d_attended, d_weights["attn.proj.w"], d_bias=d_biases["attn.proj.b"]))
        d_residual = kept(add_resident(d_hidden, d_projected))

        d_normed = kept(layernorm_resident(d_residual, d_constants[1]))
        d_wide = kept(gemm_resident(d_normed, d_weights["mlp.fc.w"], d_bias=d_biases["mlp.fc.b"]))
        # Applies the GELU while staging the down projection's operand, so it costs no launch of its own.
        d_narrow = kept(gemm_resident(d_wide, d_weights["mlp.proj.w"], activate=True,
                                      d_bias=d_biases["mlp.proj.b"]))
        return download_activations(kept(add_resident(d_residual, d_narrow)))
    finally:
        for d_tensor in d_owned:
            d_tensor.free()
        for address in d_constants:
            minitpuFree(address)


def _launch(image: int, size: int, *, args: list[int]) -> None:
    """Launches one staged image and waits for the NPU to retire it, adding that time to the NPU clock."""
    started = time.perf_counter()
    minitpuLaunchKernel(image, size, args=args)
    _NPU_CLOCK[0] += time.perf_counter() - started


def _manifest() -> dict:
    """Returns the manifest describing every compiled image, keyed by file name."""
    return json.loads((_IMAGE_DIRECTORY / "manifest.json").read_text())


def _launch_copy(*, rows: int, stride: int, source: int, destination: int) -> None:
    """Copies rows spaced `stride` bytes apart from one DMEM address to another through the compiled copy image."""
    image_bytes, _ = _compiled_image(kind="copy", rows=rows, stride=stride)
    minitpuSetDevice(UIO_NAME)
    d_image = _stage_image(image_bytes)
    _launch(d_image, len(image_bytes), args=[source, 0, destination])


def _decode_constants(entry: dict, *, length: int) -> int:
    """Writes a decode step's constants block, masking every key past the cache's length, and returns its address.

    Args:
        entry: The decode image's manifest entry, which states the cache length it walks and its constants.
        length: Keys the cache holds.

    Returns:
        The device address the image binds as its constants argument.
    """
    keys = entry["keys"]
    h_words = pack_attention_constants(initial_maximum=entry["entering"], constants=tuple(entry["stated"]),
                                       masks=decode_mask_tiles(keys=keys, length=length)).reshape(-1)
    d_address = _DECODE_CONSTANTS.get(keys)
    if d_address is None:
        d_address = minitpuMalloc(h_words.nbytes)
        _DECODE_CONSTANTS[keys] = d_address
    h_staging = minitpuHostAlloc(h_words.nbytes)
    h_staging.view.view("<u2")[: h_words.size] = h_words
    minitpuMemcpy(d_address, h_staging, h_words.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    return d_address


def _stage_image(image_bytes: bytes) -> int:
    """Makes a kernel image resident in DMEM, once, and returns its address.

    Args:
        image_bytes: The assembled image.

    Returns:
        The DMEM address it is resident at, to pass as `kbin_addr`.
    """
    resident = _RESIDENT_IMAGES.get(image_bytes)
    if resident is not None:
        return resident
    # Its own address rather than the shared staging one, which a launch does not require and which made a
    # layer restage on every one of its eighteen launches.
    address = minitpuMalloc(len(image_bytes))
    host = minitpuHostAlloc(len(image_bytes))
    host.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
    minitpuMemcpy(address, host, len(image_bytes), MemcpyKind.HostToDevice)
    _RESIDENT_IMAGES[image_bytes] = address
    return address


def _allocate_tensor(tokens: int, features: int) -> DeviceTensor:
    """Hands out a drain-shaped DMEM buffer, from the pool when one is waiting.

    Notes:
        The kernel loads its clearing tile from the head of the drain buffer, so those bytes must be zero
        before the first launch that uses it, but only once per buffer: a launch reads that tile and never
        writes it, so a pooled buffer comes back with its tile still zeroed and needs neither a malloc nor a
        copy.

    Args:
        tokens: Rows the buffer holds.
        features: Columns the buffer holds.

    Returns:
        The allocated tensor.
    """
    if tokens % _BLOCK_TOKENS:
        message = (
            f"Unable to hold {tokens} rows in DMEM. A tensor is a whole number of "
            f"{_BLOCK_TOKENS}-row blocks, which is what one launch covers."
        )
        raise ValueError(message)
    waiting = _TENSOR_POOL.get((tokens, features))
    if waiting:
        return DeviceTensor(base=waiting.pop(), tokens=tokens, features=features)

    # Allocating and zeroing both need a bound device, and a caller may reach this
    # before any launch has bound one.
    minitpuSetDevice(UIO_NAME)
    tensor = DeviceTensor(base=0, tokens=tokens, features=features)
    tensor.base = minitpuMalloc(tensor.nbytes)
    host = minitpuHostAlloc(_ZERO_TILE_BYTES)
    host.view.view("<u2")[:_ZERO_TILE_WORDS] = 0
    for block in range(tensor.blocks):
        minitpuMemcpy(tensor.d_drain(block), host, _ZERO_TILE_BYTES, MemcpyKind.HostToDevice)
    return tensor


def main() -> int:
    """Generates a prompt's continuation on the board and prints it.

    Returns:
        Zero once the continuation is printed.
    """
    parser = argparse.ArgumentParser(description="Generates a prompt's continuation with GPT-2 small on the NPU.")
    parser.add_argument("--prompt", default="The capital of France is", help="the text to continue")
    parser.add_argument("--tokens", type=int, default=8, help="tokens to generate at most")
    arguments = parser.parse_args()

    tokenizer = Tokenizer()
    layers, shared = load_gpt2_weights()
    prompt = np.asarray(tokenizer.encode(arguments.prompt), dtype=np.int64)
    started, launched = time.time(), npu_seconds()
    generated = generate(prompt, layers, shared, new_tokens=arguments.tokens)
    elapsed, on_npu = time.time() - started, npu_seconds() - launched
    print(arguments.prompt + tokenizer.decode(generated))
    print(f"[{len(prompt)} prompt tokens, {len(generated)} generated in {elapsed:.1f} s, "
          f"{on_npu:.2f} s of it on the NPU]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
