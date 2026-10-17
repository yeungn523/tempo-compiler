#!/usr/bin/env python3
"""Provides the GPT-2 host drivers, the layer references and the checks that say the device path is right.

Running this module on a board executes every check in order. Composition lives here, while the arithmetic is in
bf16.py, the hand-written schedules are in kernels.py and every GEMM image comes from the compiler through images/.
A transformer block runs resident in DMEM, every operation in it a launch reading the previous one's output.
"""

from __future__ import annotations

from collections.abc import Callable

import json
import math
import pathlib
import sys
import time

import numpy as np
from numpy.typing import NDArray

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from bf16 import (  # noqa: E402,F401  (re-exported, the operand tests importing them from here)
    MAX_ATTENTION_ROWS,
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_DEPTH,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _CORRUPTION_WORD,
    _ELEMENT_BYTES,
    _LAUNCH_REPEATS,
    _LN_EPSILON,
    _MAX_DESCRIPTOR_ROWS,
    _MODEL_WIDTH,
    _NUM_LANES,
    _ROW_BYTES,
    _TILE,
    _TILE_VREGS,
    _VREG_ELEMENTS,
    _ZERO_TILE_BYTES,
    _ZERO_TILE_WORDS,
    _on_bf16_grid,
    _single_launch_operands,
    as_bf16_words,
    assert_operands_discriminating,
    build_operands,
    from_bf16_words,
    gelu_lut_reference,
    layernorm_constants,
    pack_activations,
    pack_bias,
    pack_weights,
)
from kernels import (  # noqa: E402
    build_layernorm_rows,
    build_elementwise_add,
    build_head_copy,
    build_head_transpose,
    build_matrix_transpose,
    build_looped_gemm,
    build_softmax_packed,
    build_softmax_rows,
    single_halt,
)
from runtime import (  # noqa: E402
    MemcpyKind,
    MinitpuKernel,
    minitpuFree,
    minitpuGetAddressSpace,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
)


sys.path.insert(0, str(HERE))


sys.path.insert(0, str(HERE / "minitpu_runtime"))




_GEMM_ERROR_LIMIT: float = 0.01


_SOFTMAX_ERROR_LIMIT: float = 0.02


_BLOCK_ERROR_LIMIT: float = 0.05


_TENSOR_POOL: dict[tuple[int, int], list[int]] = {}
"""Freed tensors waiting to be handed out again, keyed by shape. A launch reads a drain buffer's clearing tile
and never writes it, so a reused buffer needs no re-zeroing, which is the copy the pool saves."""


_RESIDENT_IMAGES: dict[bytes, int] = {}
"""What each kernel base currently holds, so relaunching an unchanged image copies nothing. Sound only while
every launch in the process stages through `_stage_image`, which a program mixing in a runtime-staged kernel
would not, and the cache would then belong in the runtime instead."""


_KERNEL_CACHE: dict[tuple, tuple] = {}
"""Assembled kernels and their serialised images, keyed by the arguments that built them, serialising alone
having measured 36.8 ms of a 47.4 ms fully resident launch. The key is the arguments rather than the kernel:
`id` is reused once CPython collects the object, which handed a dead kernel's image to a later one, and a frozen
dataclass cannot be memoised on."""


_IMAGE_DIRECTORY: pathlib.Path = HERE / "images"
"""Directory holding the compiled GEMM images and the manifest describing them, which a host writes through
`tools/build_gpt2_images.py`."""


_UIO_NAME: str = "minitpu"
"""UIO node every image launches on."""


_COMPILED_GEMMS: dict[tuple[int, int, bool], bytes] = {}
"""Compiled GEMM images keyed by contraction depth, output columns and activation, each read from disk once."""


_ATTENTION_MASKS: dict[tuple, DeviceTensor] = {}
"""Additive attention masks, keyed by shape. A mask is a constant of a shape rather than of a layer, so one
serves every layer and is built once."""


_GROUPED_OPERANDS: dict[tuple, tuple] = {}
"""Attention's per-head operands, keyed by shape. Every layer's scatter overwrites them, so one set serves every
layer and is zeroed once."""


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
    def shape(self) -> tuple[int, int]:
        return (self.tokens, self.features)

    def operand(self, block: int) -> int:
        """The address a launch binds as its activation argument for one block."""
        return self._d_parent.operand(block) + self._offset


def layernorm_reference(
    rows: NDArray[np.float32], gamma: NDArray[np.float32], beta: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Returns the LayerNorm a BF16 device computes, on the host.

    Notes:
        Rounds where the device rounds: the mean and variance are BF16 sums scaled by a BF16 1/width,
        and the affine is two BF16 operations. What is left is the rsqrt table's interpolation.
    """
    values = _on_bf16_grid(values=rows)
    inverse = _on_bf16_grid(values=np.float32(1.0 / _MODEL_WIDTH))
    mean = _on_bf16_grid(values=values.sum(axis=1, keepdims=True, dtype=np.float32) * inverse)
    centred = _on_bf16_grid(values=values - mean)
    variance = _on_bf16_grid(
        values=_on_bf16_grid(values=centred * centred).sum(axis=1, keepdims=True, dtype=np.float32)
        * inverse)
    scale = _on_bf16_grid(values=1.0 / np.sqrt(_on_bf16_grid(values=variance + _LN_EPSILON)))
    return _on_bf16_grid(values=_on_bf16_grid(values=centred * scale) * _on_bf16_grid(values=gamma)
                         + _on_bf16_grid(values=beta))


def softmax_reference(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns the row-wise softmax a BF16 device computes, on the host.

    Notes:
        Quantises at each step the device quantises at: the subtraction, the exponential and the
        product all round to BF16, and the reciprocal is the SFU's own. What is left over is the
        table's interpolation error, which is what a check's bound has to cover.

    Args:
        scores: One row per reduction, each `_VREG_ELEMENTS` wide.

    Returns:
        The float32 softmax of each row.
    """
    values = _on_bf16_grid(values=scores)
    shifted = _on_bf16_grid(values=values - values.max(axis=1, keepdims=True))
    weights = _on_bf16_grid(values=np.exp(shifted, dtype=np.float32))
    totals = _on_bf16_grid(values=weights.sum(axis=1, keepdims=True, dtype=np.float32))
    return _on_bf16_grid(values=weights * _on_bf16_grid(values=1.0 / totals))


def upload_weights(h_weights: NDArray[np.float32]) -> ResidentWeights:
    """Packs a weight matrix and leaves it in DMEM for repeated launches.

    Args:
        h_weights: [contraction, columns] host operand, rows being the contraction.

    Returns:
        A handle `full_gemm` accepts in place of the array.
    """
    words = pack_weights(weights=_on_bf16_grid(values=h_weights.astype(np.float32)))
    minitpuSetDevice(single_halt.uio_name)
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
    minitpuSetDevice(single_halt.uio_name)
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
    """Uploads one LayerNorm's packed constants and returns their DMEM address.

    Args:
        h_gamma: Per-feature scale, on the host.
        h_beta: Per-feature shift, on the host.

    Returns:
        The device address the kernel binds as its constants argument.
    """
    h_constants = layernorm_constants(gamma=h_gamma, beta=h_beta)
    h_staging = minitpuHostAlloc(h_constants.nbytes)
    h_staging.view.view("<u2")[: h_constants.size] = h_constants
    d_address = minitpuMalloc(h_constants.nbytes)
    minitpuMemcpy(d_address, h_staging, h_constants.nbytes, MemcpyKind.HostToDevice)
    minitpuHostReset()
    return d_address


def layernorm_resident(d_rows: DeviceTensor, d_constants: int, *,
                       d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Normalises a resident tensor in place of the layout, leaving it resident.

    Notes:
        The kernel reads and writes in the layout a GEMM operand uses, so a normalised tensor feeds the next
        projection where it lies rather than returning to PS DDR to be re-packed.

    Args:
        d_rows: [tokens, model] resident rows.
        d_constants: DMEM address returned by `upload_layernorm_constants`.
        d_out: Where to write. Allocated when omitted.

    Returns:
        The normalised rows, resident.

    Raises:
        ValueError: If the rows are not the model width the kernel is built for.
    """
    if d_rows.features != _MODEL_WIDTH:
        message = (
            f"Unable to normalise rows of {d_rows.features}. The kernel's tile count is "
            f"derived from {_MODEL_WIDTH}, so a row must be exactly that wide."
        )
        raise ValueError(message)
    if d_out is None:
        d_out = _allocate_tensor(d_rows.tokens, d_rows.features)

    kernel, image_bytes = _kernel_and_image(
        ("layernorm_packed", _BLOCK_TOKENS),
        lambda: build_layernorm_rows(rows=_BLOCK_TOKENS, packed=True))
    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)
    for block in range(d_rows.blocks):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_rows.operand(block), d_constants, d_out.operand(block)])
    return d_out


def transpose_2d(h_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Transposes a whole [rows, columns] array on the device.

    Notes:
        A tile of a row-major array is 16 rows at stride `columns / NUM_LANES` rather than 16 consecutive
        ones, which is the only descriptor stride other than one in this package. Tile (i, j) becomes tile
        (j, i) by pointer arithmetic: one launch per source tile column walks that column at the load stride
        and lands its tiles consecutively in the destination row they belong to.

    Args:
        h_matrix: [rows, columns] host array, both a whole number of `_TILE`.

    Returns:
        Its [columns, rows] transpose.

    Raises:
        ValueError: If either dimension is not a whole number of tiles.
    """
    rows, columns = h_matrix.shape
    if rows % _TILE or columns % _TILE:
        message = (
            f"Unable to transpose a [{rows},{columns}] array. The staging tile is "
            f"{_TILE}x{_TILE}, so both dimensions must be a whole number of {_TILE}."
        )
        raise ValueError(message)
    down, across = rows // _TILE, columns // _TILE

    kernel, image_bytes = _kernel_and_image(
        ("transpose2d", down, across, columns // _NUM_LANES, rows // _NUM_LANES),
        lambda: build_matrix_transpose(down=down, across=across,
                                       load_stride=columns // _NUM_LANES,
                                       store_stride=rows // _NUM_LANES))
    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    d_owned: list[int] = []
    try:
        d_image = _stage_image(image_bytes)
        words = as_bf16_words(values=_on_bf16_grid(values=h_matrix)).reshape(-1)
        h_in = minitpuHostAlloc(words.nbytes)
        h_in.view.view("<u2")[: words.size] = words
        d_in = minitpuMalloc(words.nbytes)
        d_owned.append(d_in)
        minitpuMemcpy(d_in, h_in, words.nbytes, MemcpyKind.HostToDevice)
        h_out = minitpuHostAlloc(words.nbytes)
        d_out = minitpuMalloc(words.nbytes)
        d_owned.append(d_out)

        # One launch for the whole matrix, the tile-column walk being the kernel's outer loop rather than the
        # host's, which at GPT-2's width was 48 launches of a 32-row transpose.
        minitpuLaunchKernel(d_image, len(image_bytes),
                            args=[d_in, 0, d_out])
        minitpuMemcpy(h_out, d_out, words.nbytes, MemcpyKind.DeviceToHost)
        return from_bf16_words(
            words=h_out.view.view("<u2")[: words.size]).reshape(columns, rows)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()


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
        ValueError: If the operands disagree in shape.
    """
    if d_left.shape != d_right.shape:
        message = (
            f"Unable to add a {d_left.shape} tensor to a {d_right.shape} one. "
            f"Elementwise addition needs the same shape."
        )
        raise ValueError(message)
    if d_out is None:
        d_out = _allocate_tensor(d_left.tokens, d_left.features)

    rows_per_block = _BLOCK_TOKENS * d_left.features // _NUM_LANES
    if rows_per_block % _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to add tensors of {d_left.features} features. A block is "
            f"{rows_per_block} VMEM rows and the kernel moves {_MAX_DESCRIPTOR_ROWS} "
            f"per iteration, so it must divide evenly."
        )
        raise ValueError(message)
    chunks = rows_per_block // _MAX_DESCRIPTOR_ROWS

    kernel, image_bytes = _kernel_and_image(
        ("add", chunks), lambda: build_elementwise_add(chunks=chunks))
    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)
    for block in range(d_left.blocks):
        minitpuLaunchKernel(
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
    """
    if d_bias.depth != 1 or d_bias.columns != d_left.features:
        raise ValueError(
            f"Unable to broadcast bias {d_bias.shape} over {d_left.shape}. The "
            "bias must be one row with the tensor's feature width."
        )
    if d_out is None:
        d_out = _allocate_tensor(d_left.tokens, d_left.features)

    rows_per_block = _BLOCK_TOKENS * d_left.features // _NUM_LANES
    if rows_per_block % _MAX_DESCRIPTOR_ROWS:
        raise ValueError(
            f"Unable to add a bias over {d_left.features} features. A block is "
            f"{rows_per_block} VMEM rows and the kernel moves "
            f"{_MAX_DESCRIPTOR_ROWS} per iteration."
        )
    chunks = rows_per_block // _MAX_DESCRIPTOR_ROWS
    kernel, image_bytes = _kernel_and_image(
        ("add_bias", chunks),
        lambda: build_elementwise_add(chunks=chunks, broadcast_right=True))
    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)
    for block in range(d_left.blocks):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_left.operand(block), d_bias.address, d_out.operand(block)])
    return d_out


def gemm_resident(d_activations: DeviceTensor, d_weights: ResidentWeights, *,
                  activate: bool = False, d_out: DeviceTensor | None = None) -> DeviceTensor:
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

    image_bytes = _compiled_gemm(depth=depth, columns=columns, activate=activate)

    if d_out is None:
        d_out = _allocate_tensor(tokens, columns)

    minitpuSetDevice(_UIO_NAME)
    d_image = _stage_image(image_bytes)

    for block in range(tokens // _BLOCK_TOKENS):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_activations.operand(block), d_weights.address, d_out.d_drain(block)])
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

    minitpuSetDevice(_UIO_NAME)
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

            minitpuLaunchKernel(d_image, len(image_bytes),
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


def softmax_rows(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Applies a row-wise softmax on the device.

    Notes:
        One row is one VREG, so the row width is fixed at `_VREG_ELEMENTS`, and the rows are launched
        in chunks of whatever one DMA descriptor covers.

    Args:
        scores: One row per reduction, each `_VREG_ELEMENTS` wide.

    Returns:
        The float32 softmax of each row.

    Raises:
        ValueError: If the rows are not the width one VREG reduces over.
    """
    rows, width = scores.shape
    if width != _VREG_ELEMENTS:
        message = (
            f"Unable to softmax rows of {width}. vredsum and vredmax reduce over a whole VREG, so a "
            f"row must be exactly {_VREG_ELEMENTS} wide."
        )
        raise ValueError(message)
    chunk = _MAX_DESCRIPTOR_ROWS // _TILE_VREGS
    whole, tail = divmod(rows, chunk)

    minitpuSetDevice(single_halt.uio_name)
    minitpuHostReset()
    d_owned: list[int] = []
    h_result = np.empty_like(scores, dtype=np.float32)
    try:
        span = rows * width
        h_scores = minitpuHostAlloc(span * _ELEMENT_BYTES)
        d_scores = minitpuMalloc(span * _ELEMENT_BYTES)
        d_owned.append(d_scores)
        h_out = minitpuHostAlloc(span * _ELEMENT_BYTES)
        d_out = minitpuMalloc(span * _ELEMENT_BYTES)
        d_owned.append(d_out)

        words = as_bf16_words(values=scores).reshape(-1)
        h_scores.view.view("<u2")[: words.size] = words
        minitpuMemcpy(d_scores, h_scores, words.nbytes, MemcpyKind.HostToDevice)

        # One launch streams every whole chunk, the kernel advancing its own pointers and its inner loop naming
        # its AGU level so the chunk loop does not steal the row index. A ragged tail needs an image of its own,
        # the inner trip count being baked in, which is one more launch rather than one each.
        stride = chunk * width * _ELEMENT_BYTES
        if whole:
            _, image_bytes = _kernel_and_image(
                ("softmax", chunk, whole),
                lambda: build_softmax_rows(chunk, chunks=whole))
            d_image = _stage_image(image_bytes)
            minitpuLaunchKernel(d_image, len(image_bytes),
                                args=[d_scores, 0, d_out])
        if tail:
            _, image_bytes = _kernel_and_image(
                ("softmax", tail, 1), lambda: build_softmax_rows(tail, chunks=1))
            d_image = _stage_image(image_bytes)
            minitpuLaunchKernel(d_image, len(image_bytes),
                                args=[d_scores + whole * stride, 0,
                                      d_out + whole * stride])

        minitpuMemcpy(h_out, d_out, words.nbytes, MemcpyKind.DeviceToHost)
        h_result[:] = from_bf16_words(
            words=h_out.view.view("<u2")[: words.size]).reshape(scores.shape)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()
    return h_result


def layernorm_rows(
    rows: NDArray[np.float32], gamma: NDArray[np.float32], beta: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Applies LayerNorm to every hidden row on the device, in one launch.

    Args:
        rows: One hidden row per token, each `_MODEL_WIDTH` wide.
        gamma: Per-feature scale.
        beta: Per-feature shift.

    Returns:
        The normalised rows.

    Raises:
        ValueError: If the rows are not the model width.
    """
    count, width = rows.shape
    if width != _MODEL_WIDTH:
        message = (
            f"Unable to normalise rows of {width}. The kernel's tile count is derived from "
            f"{_MODEL_WIDTH}, so a row must be exactly that wide."
        )
        raise ValueError(message)

    kernel, image_bytes = _kernel_and_image(
        ("layernorm", count), lambda: build_layernorm_rows(rows=count))
    constants = layernorm_constants(gamma=gamma, beta=beta)
    words = as_bf16_words(values=rows).reshape(-1)

    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    d_owned: list[int] = []
    try:
        d_image = _stage_image(image_bytes)

        input_host = minitpuHostAlloc(words.nbytes)
        input_host.view.view("<u2")[: words.size] = words
        input_base = minitpuMalloc(words.nbytes)
        d_owned.append(input_base)
        minitpuMemcpy(input_base, input_host, words.nbytes, MemcpyKind.HostToDevice)

        const_host = minitpuHostAlloc(constants.nbytes)
        const_host.view.view("<u2")[: constants.size] = constants
        const_base = minitpuMalloc(constants.nbytes)
        d_owned.append(const_base)
        minitpuMemcpy(const_base, const_host, constants.nbytes, MemcpyKind.HostToDevice)

        h_out = minitpuHostAlloc(words.nbytes)
        d_out = minitpuMalloc(words.nbytes)
        d_owned.append(d_out)

        minitpuLaunchKernel(d_image, len(image_bytes),
                            args=[input_base, const_base, d_out])
        minitpuMemcpy(h_out, d_out, words.nbytes, MemcpyKind.DeviceToHost)
        return from_bf16_words(words=h_out.view.view("<u2")[: words.size]).reshape(count, width)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()


def transformer_block(hidden: NDArray[np.float32], weights: dict, *, heads: int = 12) -> NDArray[np.float32]:
    """Runs one GPT-2 transformer block on the device, keeping every intermediate resident in DMEM.

    Notes:
        Both LayerNorms, all four projections, every bias and residual add and the attention run as launches
        that read the previous launch's output where it lies, so the block crosses the host boundary once on
        entry and once on exit. The four projections launch compiled images. Attention's 1/sqrt(head) rides
        the query columns of the QKV projection, folded into its weights and bias before upload.

    Args:
        hidden: [tokens, model] block input, a whole number of token blocks and at most `MAX_ATTENTION_ROWS`.
        weights: The block's parameters, keyed as GPT-2 names them: ``ln_1.g``, ``ln_1.b``,
            ``attn.qkv.w``, ``attn.qkv.b``, ``attn.proj.w``, ``attn.proj.b``, ``ln_2.g``,
            ``ln_2.b``, ``mlp.fc.w``, ``mlp.fc.b``, ``mlp.proj.w``, ``mlp.proj.b``.
        heads: Attention heads the model width splits into.

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
        d_fused = kept(add_bias_resident(kept(gemm_resident(d_normed, d_weights["attn.qkv.w"])),
                                         d_biases["attn.qkv.b"]))
        d_attended = kept(attention(d_fused, heads=heads, causal=False))
        d_projected = kept(add_bias_resident(kept(gemm_resident(d_attended, d_weights["attn.proj.w"])),
                                             d_biases["attn.proj.b"]))
        d_residual = kept(add_resident(d_hidden, d_projected))

        d_normed = kept(layernorm_resident(d_residual, d_constants[1]))
        d_wide = kept(add_bias_resident(kept(gemm_resident(d_normed, d_weights["mlp.fc.w"])),
                                        d_biases["mlp.fc.b"]))
        # Applies the GELU while staging the down projection's operand, so it costs no launch of its own.
        d_narrow = kept(add_bias_resident(kept(gemm_resident(d_wide, d_weights["mlp.proj.w"], activate=True)),
                                          d_biases["mlp.proj.b"]))
        return download_activations(kept(add_resident(d_residual, d_narrow)))
    finally:
        for d_tensor in d_owned:
            d_tensor.free()
        for address in d_constants:
            minitpuFree(address)


def transformer_block_reference(hidden: NDArray[np.float32], weights: dict) -> NDArray[np.float32]:
    """Returns the block output a BF16 device computes, on the host."""
    normalised = layernorm_reference(rows=hidden, gamma=weights["ln_1.g"], beta=weights["ln_1.b"])
    fused = (normalised.astype(np.float64) @ weights["attn.qkv.w"].astype(np.float64)
             + weights["attn.qkv.b"]).astype(np.float32)
    query, key, value = np.split(fused, 3, axis=1)
    heads = 12
    head = hidden.shape[1] // heads
    gathered = np.concatenate(
        [attention_reference(query=np.ascontiguousarray(query[:, i * head:(i + 1) * head]),
                             key=np.ascontiguousarray(key[:, i * head:(i + 1) * head]),
                             value=np.ascontiguousarray(value[:, i * head:(i + 1) * head]))
         for i in range(heads)], axis=1)
    attended = (_on_bf16_grid(values=gathered).astype(np.float64)
                @ weights["attn.proj.w"].astype(np.float64) + weights["attn.proj.b"]).astype(np.float32)
    residual = _on_bf16_grid(values=hidden + attended)

    normalised = layernorm_reference(rows=residual, gamma=weights["ln_2.g"], beta=weights["ln_2.b"])
    expanded = (normalised.astype(np.float64) @ weights["mlp.fc.w"].astype(np.float64)
                + weights["mlp.fc.b"]).astype(np.float32)
    activated = from_bf16_words(words=gelu_lut_reference(words=as_bf16_words(values=expanded)))
    contracted = (activated.astype(np.float64) @ weights["mlp.proj.w"].astype(np.float64)
                  + weights["mlp.proj.b"]).astype(np.float32)
    return _on_bf16_grid(values=residual + contracted)


def attention(d_fused: DeviceTensor, *, heads: int = 12, causal: bool = True,
                      sequences: int = 1, valid: int | None = None) -> DeviceTensor:
    """Attention with each head an independent product, not a diagonal block.

    Notes:
        The block-diagonal form runs every head against an operand that is mostly zeros, costing `heads` times
        the multiplies the problem has. This runs the heads as that many problems in one launch instead, so the
        zeros are gone and the launch count is no worse. Nothing downstream changes: a grouped launch drains
        problem after problem in the ordering the packed score array already has. Only `d_keys` changes, holding
        one transposed key matrix per head end to end rather than one block-diagonal.

    Args:
        d_fused: [rows, 3 * model] fused QKV projection, resident.
        heads: Attention heads.
        causal: Whether a query may only attend at or before its own position.
        sequences: Independent sequences down the token axis, kept apart by the mask.
        valid: Real rows, where `d_fused` may be padded to a launch's block.

    Returns:
        The [rows, model] attention output, resident.

    Raises:
        ValueError: If the sequences do not divide the rows evenly.
    """
    rows, fused_width = d_fused.shape
    model = fused_width // 3
    head = model // heads
    span = -(-rows // _VREG_ELEMENTS) * _VREG_ELEMENTS
    valid = rows if valid is None else valid
    sequence = _sequence_length(rows=rows, valid=valid,
                                sequences=sequences)

    model_tiles = model // _TILE
    d_query = DeviceTensorSlice(d_fused, 0, model)
    d_key = DeviceTensorSlice(d_fused, model_tiles, model)
    d_value = DeviceTensorSlice(d_fused, 2 * model_tiles, model)

    d_keys, d_values = _grouped_operands(rows, heads, head, span)
    d_mask = _attention_mask(rows, heads, causal=causal, sequence=sequence,
                             valid=valid, span=span)

    head_tiles = head // _TILE
    down = _BLOCK_TOKENS // _TILE
    blocks = rows // _BLOCK_TOKENS

    # One [head_dim, span] transpose per head, scattered as the block-diagonal's is but against a head's depth
    # rather than the model's, so a head's tiles land `head_dim` apart and the next begins where it ended.
    scatter_key, key_image = _kernel_and_image(
        ("head_transpose", heads, head_tiles, down, head, span // _TILE),
        # Grouped: no row band, so the feature-tile loop's walk is taken back.
        lambda: build_head_transpose(
            heads=heads, head_tiles=head_tiles, down=down, depth=head,
            head_advance=span // _TILE * head - head_tiles * _TILE))
    minitpuSetDevice(scatter_key.uio_name)
    d_keys_image = _stage_image(key_image)
    for block in range(blocks):
        minitpuLaunchKernel(
            d_keys_image, len(key_image),
            args=[d_key.operand(block), 0,
                  d_keys.address + block * down * head * _ROW_BYTES])

    # V, one [span, head_dim] matrix per head, untransposed. `head_advance` is zero:
    # the inner loop already walked the whole of this head's matrix.
    scatter_value, value_image = _kernel_and_image(
        ("head_copy", heads, head_tiles, _BLOCK_TOKENS, span),
        lambda: build_head_copy(heads=heads, head_tiles=head_tiles,
                                     tokens=_BLOCK_TOKENS, depth=span,
                                     head_advance=0))
    d_values_image = _stage_image(value_image)
    for block in range(blocks):
        minitpuLaunchKernel(
            d_values_image, len(value_image),
            args=[d_value.operand(block), 0,
                  d_values.address + block * _BLOCK_TOKENS * _ROW_BYTES])

    d_scores = _grouped_product(d_query, d_keys, heads=heads, rows=rows,
                                columns=span, name="grouped_scores")
    d_scores = add_resident(d_scores, d_mask, d_out=d_scores)

    softmax, softmax_image = _kernel_and_image(
        ("softmax_packed", _BLOCK_TOKENS, heads, span),
        lambda: build_softmax_packed(tokens=_BLOCK_TOKENS, heads=heads, span=span))
    minitpuSetDevice(softmax.uio_name)
    d_softmax_image = _stage_image(softmax_image)
    d_probabilities = _allocate_tensor(rows, heads * span)
    for block in range(blocks):
        minitpuLaunchKernel(
            d_softmax_image, len(softmax_image),
            args=[d_scores.operand(block), 0, d_probabilities.operand(block)])
    d_scores.free()

    d_gathered = _grouped_product(d_probabilities, d_values, heads=heads, rows=rows,
                                  columns=head, name="grouped_gather")
    d_probabilities.free()
    return d_gathered


def prefill_attention(
    query: NDArray[np.float32], key: NDArray[np.float32], value: NDArray[np.float32],
    *, heads: int = 12, causal: bool = True, scale: float | None = None
) -> NDArray[np.float32]:
    """Runs prefill attention for EVERY head on the device, in three launches.

    Notes:
        Same batching as `decode_attention`: head h's transposed keys occupy block (h, h) of a block-diagonal
        operand, so one product gives every head's score matrix side by side and no head sees another's keys.
        Splitting heads is addressing rather than movement, both reshapes being contiguous views over the same
        bytes, so only the key transpose costs anything. The causal mask is the one arithmetic step still on the
        host, since adding it needs the scores to stay in VMEM between the product and the softmax.

    Args:
        query: [tokens, model] queries for all heads, unscaled.
        key: [tokens, model] keys for all heads.
        value: [tokens, model] values for all heads.
        heads: Attention heads.
        causal: Whether a query may only attend to keys at or before its own position.
        scale: What to multiply the queries by. Defaults to 1/sqrt(head_dim). Pass
            1.0 when the caller has already folded the scale into the projection
            that produced these queries, as `ResidentModel` does, applying it twice
            is silent and looks like a plausible attention distribution.

    Returns:
        The [tokens, model] attention output, heads concatenated.

    Raises:
        ValueError: If the token count exceeds the keys one VREG reduces over.
    """
    tokens, model = query.shape
    head = model // heads
    if not 0 < tokens <= _VREG_ELEMENTS:
        message = (
            f"Unable to attend over {tokens} tokens. A score row is one VREG, which `vredmax` and "
            f"`vredsum` reduce over, so it holds at most {_VREG_ELEMENTS} keys."
        )
        raise ValueError(message)

    span = _VREG_ELEMENTS
    if scale is None:
        scale = 1.0 / math.sqrt(head)
    scaled = (query * scale).astype(np.float32)

    # The transposed keys come from the device. Their block-diagonal placement is still host bookkeeping,
    # zeros off the diagonal so no head sees another's keys, but the transpose has no software route.
    h_key_t = transpose_2d(_pad_rows_to_tile(key))[:, :tokens]
    key_blocks = np.zeros((model, heads * span), dtype=np.float32)
    value_blocks = np.zeros((heads * span, model), dtype=np.float32)
    for h in range(heads):
        columns = slice(h * head, (h + 1) * head)
        key_blocks[columns, h * span:h * span + tokens] = h_key_t[columns]
        value_blocks[h * span:h * span + tokens, columns] = value[:, columns]

    scores = full_gemm(activations=_pad_tokens_for_launch(_on_bf16_grid(values=scaled)),
                       weights=_on_bf16_grid(values=key_blocks))[:tokens]

    # [tokens, heads*span] over the same bytes as [tokens*heads, span]: a view, not a copy.
    rows = scores.reshape(tokens * heads, span)
    if tokens < span:
        # Keys past the context do not exist. Large and negative rather than -inf, because the
        # kernel subtracts the row maximum before the exponential and inf - inf is a NaN.
        rows[:, tokens:] = -1.0e30
    if causal:
        positions = np.repeat(np.arange(tokens), heads)[:, None]
        rows[:, :tokens][np.arange(tokens)[None, :] > positions] = -1.0e30

    probabilities = softmax_rows(scores=_on_bf16_grid(values=rows))
    gathered = full_gemm(
        activations=_pad_tokens_for_launch(
            _on_bf16_grid(values=probabilities.reshape(tokens, heads * span))),
        weights=_on_bf16_grid(values=value_blocks))
    return gathered[:tokens]


def decode_attention(
    query: NDArray[np.float32], keys: NDArray[np.float32], values: NDArray[np.float32],
    *, scale: float | None = None
) -> NDArray[np.float32]:
    """Runs one decode step's attention for EVERY head on the device, in three launches.

    Notes:
        A decode step has one query row per head, so all heads share one launch rather than paying a launch
        each: head h's transposed keys sit at block (h, h) of a block-diagonal operand, everything off the
        diagonal is zero, and no head sees another's keys. That wastes eleven twelfths of the multiplies, which
        is the right trade while the device is idle for most of a token and the launch is what costs. Contexts
        shorter than a full VREG need their unused score columns pushed to zero probability, since each head's
        score row is one whole VREG the softmax reduces over.

    Args:
        query: [heads, head] one query row per head, unscaled.
        keys: [heads, context, head] cached keys.
        values: [heads, context, head] cached values.
        scale: What to multiply the queries by. Defaults to 1/sqrt(head_dim); pass
            1.0 when the projection already carries it.

    Returns:
        The [heads, head] attention output, one row per head.

    Raises:
        ValueError: If the context does not fit the VREG a score row reduces over.
    """
    heads, head = query.shape
    context = keys.shape[1]
    if not 0 < context <= _VREG_ELEMENTS:
        message = (
            f"Unable to attend over a context of {context}. A score row is one VREG, which "
            f"`vredmax` and `vredsum` reduce over, so it holds at most {_VREG_ELEMENTS} keys."
        )
        raise ValueError(message)

    width = heads * head                       # the model width, for GPT-2 exactly 768
    span = heads * _VREG_ELEMENTS

    if scale is None:
        scale = 1.0 / math.sqrt(head)
    scaled = (query * scale).astype(np.float32)

    # Scores are [1, width] against a block-diagonal transpose of K, each head landing in its own column band.
    # One transpose covers every head: the stacked caches come back transposed, and each head's block is a
    # column range of that.
    h_stacked = _pad_rows_to_tile(
        np.ascontiguousarray(keys.reshape(heads * context, head)))
    h_keys_t = transpose_2d(h_stacked)
    key_blocks = np.zeros((width, span), dtype=np.float32)
    for h in range(heads):
        block = h_keys_t[:, h * context:(h + 1) * context]
        key_blocks[h * head:(h + 1) * head,
                   h * _VREG_ELEMENTS:h * _VREG_ELEMENTS + context] = block
    scores = full_gemm(
        activations=_pad_tokens_for_launch(scaled.reshape(1, width)),
        weights=_on_bf16_grid(values=key_blocks))[0].reshape(heads, _VREG_ELEMENTS)

    if context < _VREG_ELEMENTS:
        # Unused columns must not win probability mass. Large and negative rather than -inf: the
        # kernel subtracts the row maximum before the exponential, and inf - inf is a NaN.
        scores[:, context:] = -1.0e30

    probabilities = softmax_rows(scores=_on_bf16_grid(values=scores))

    # Values: [1, heads * VREG] @ blockdiag(V) -> [1, width], concatenated head outputs.
    value_blocks = np.zeros((span, width), dtype=np.float32)
    for h in range(heads):
        value_blocks[h * _VREG_ELEMENTS:h * _VREG_ELEMENTS + context,
                     h * head:(h + 1) * head] = values[h]
    gathered = full_gemm(
        activations=_pad_tokens_for_launch(
            _on_bf16_grid(values=probabilities.reshape(1, span))),
        weights=_on_bf16_grid(values=value_blocks))
    return gathered[0].reshape(heads, head)


def attention_reference(
    query: NDArray[np.float32], key: NDArray[np.float32], value: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Returns the attention output a BF16 device computes, on the host."""
    head = query.shape[1]
    scaled = _on_bf16_grid(values=(query / math.sqrt(head)).astype(np.float32))
    scores = _on_bf16_grid(values=(scaled.astype(np.float64) @ _on_bf16_grid(values=key).T
                                   .astype(np.float64)).astype(np.float32))
    weights = softmax_reference(scores=scores)
    return (weights.astype(np.float64) @ _on_bf16_grid(values=value).astype(np.float64)).astype(np.float32)


def check_kernel_launch(kernel: MinitpuKernel) -> None:
    """Verifies the LAUNCH path by staging a resident kernel image and starting it from the host.

    Notes:
        The transfer reaches device memory through the copy engine's alias of the kernel aperture, while the
        loader reads the staged image back through the MINITPU's own device-memory port. Only a launch crosses
        both routes, so an aperture whose two addresses resolve to different DRAM surfaces fails here rather
        than in a workload result. Repeating the launch requires the command processor to return to idle.

    Args:
        kernel: The assembled program whose image is staged and started.

    Raises:
        RuntimeError: If the loader flags an error and the command processor abandons the launch.
        TimeoutError: If a command fails to retire, which is how a core that never halts presents.
    """
    image_bytes = _image_of(kernel)
    address_space = minitpuGetAddressSpace()

    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    try:
        staging = minitpuHostAlloc(len(image_bytes))
        staging.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
        minitpuMemcpy(
            dst=address_space.kernel_base,
            src=staging,
            nbytes=len(image_bytes),
            kind=MemcpyKind.HostToDevice,
        )

        for _ in range(_LAUNCH_REPEATS):
            minitpuLaunchKernel(kbin_addr=address_space.kernel_base, kbin_size=len(image_bytes))
    finally:
        minitpuHostReset()


def check_copy_roundtrip(words: NDArray[np.uint32], *, uio_name: str = "minitpu") -> None:
    """Verifies that a kernel's argument words cross to device memory and return byte for byte unchanged.

    Notes:
        The words travel from the pinned PS-DDR host buffer to the DMEM device aperture and back, so the check
        covers minitpuMemcpy in both directions over the exact payload a kernel is given. Both transfers translate
        the device pointer the same way, so a returned payload establishes transport fidelity rather than the
        address the words occupy in DMEM.

    Args:
        words: The BF16 container words to send through device memory.
        uio_name: The name of the board's UIO device node.

    Raises:
        AssertionError: If the words the device returns differ from the words sent to it.
    """
    minitpuSetDevice(uio_name)
    minitpuHostReset()
    device_address = None
    try:
        source = minitpuHostAlloc(words.nbytes)
        destination = minitpuHostAlloc(words.nbytes)
        source.view.view("<u2")[: words.size] = words
        device_address = minitpuMalloc(words.nbytes)
        minitpuMemcpy(dst=device_address, src=source, nbytes=words.nbytes, kind=MemcpyKind.HostToDevice)

        # Writes the corruption word so a transfer that moves nothing fails. The udmabuf outlives the process and
        # every run reuses its offsets, so an untouched destination still holds the last run's words. The flush
        # carries the word past the cache lines the read-back invalidates.
        destination.view.view("<u2")[: words.size] = _CORRUPTION_WORD & 0xFFFF
        destination.sync_to_device()

        minitpuMemcpy(dst=destination, src=device_address, nbytes=words.nbytes, kind=MemcpyKind.DeviceToHost)
        np.testing.assert_array_equal(
            destination.view.view("<u2")[: words.size], words)
    finally:
        if device_address is not None:
            minitpuFree(device_address)
        minitpuHostReset()


def run_check(name: str, check: Callable[[], str]) -> bool:
    """Runs one check and reports whether it passed.

    Args:
        name: The label the report line carries.
        check: The callable performing the check, which returns the detail its report line carries.

    Returns:
        Whether the check passed.
    """
    try:
        detail = check()
    except Exception as error:  # noqa: BLE001
        print(f"[FAIL] {name}: {error}")
        return False
    print(f"[PASS] {name}: {detail}")
    return True


def check_halt() -> str:
    """Stages the shortest valid program and starts it repeatedly.

    Returns:
        The detail the report line carries.

    Raises:
        RuntimeError: If the loader flags an error and the command processor abandons the launch.
        TimeoutError: If a command fails to retire, which is how a core that never halts presents.
    """
    check_kernel_launch(kernel=single_halt)
    return f"staged and launched {_LAUNCH_REPEATS} times"


def check_copy_argument() -> str:
    """Sends one GEMM argument through device memory and reads it back.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the payload read back differs from the payload sent.
    """
    gemm_input = _single_launch_operands()[3]
    check_copy_roundtrip(words=gemm_input)
    return f"{gemm_input.nbytes}-byte GEMM argument"


def check_full_gemm(*, tokens: int = 32, depth: int = 768, columns: int = 3072) -> str:
    """Runs a product at model dimensions through its compiled image and verifies it against a host reference.

    Notes:
        Runs two operand sets, either alone being blind. The structured set is exactly representable and pins
        the kernel to the host reference bit for bit, but its token rows are identical and its weights constant
        along the contraction, so a kernel confusing one token or one contraction block for another still
        passes. The varying set differs along every axis, which makes those confusions observable, and carries
        a relative bound because its BF16 partial sums cost exactness.

    Args:
        tokens: Rows the product covers, which must be a whole number of token blocks.
        depth: The contraction, which a compiled image in the manifest must cover together with `columns`.
        columns: Output features, which must be a whole number of tiles.

    Returns:
        The detail the report line carries, naming the shapes, the launch count and the measured time.

    Raises:
        AssertionError: If the compiled image disagrees with the host reference on either operand set.
    """
    launches = tokens // _BLOCK_TOKENS

    left, right = build_operands(tokens, depth, columns, kind="structured", seed=0)

    started = time.perf_counter()
    actual = full_gemm(activations=left, weights=right)
    elapsed = time.perf_counter() - started
    np.testing.assert_allclose(actual, left @ right, rtol=0, atol=0)

    # build_operands runs assert_operands_discriminating on this pair, so a set that stopped
    # varying along an axis fails here rather than passing the check below for free.
    varying_left, varying_right = build_operands(tokens, depth, columns, kind="varying", seed=0)
    varying_actual = full_gemm(activations=varying_left, weights=varying_right)
    expected = varying_left.astype(np.float64) @ varying_right.astype(np.float64)
    error = float(np.linalg.norm(varying_actual - expected) / np.linalg.norm(expected))
    if error >= _GEMM_ERROR_LIMIT:
        message = (
            f"Unable to verify the product against the host reference on operands that vary along "
            f"every axis. The relative error must stay below {_GEMM_ERROR_LIMIT:.0%}, which is what "
            f"the BF16 partial sums explain, but got {error:.3%}. The structured operands above "
            f"passed, so the kernel agrees on a product whose token rows are identical and whose "
            f"weights are constant along the contraction, and disagrees once they are not."
        )
        raise AssertionError(message)

    return (
        f"[{tokens},{depth}] @ [{depth},{columns}] compiled, {2 * launches} launches in {elapsed:.1f} s "
        f"({1e6 * elapsed / launches:.0f} us each), varying-operand relative error {error:.2%}"
    )


def check_elementwise_add(*, tokens: int = 32, features: int = 768) -> str:
    """Adds two DMEM tensors, and one tensor and a broadcast bias row, on the device and compares against the host.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If either device sum is not the host's, exactly.
    """
    rng = np.random.default_rng(53)
    h_left = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    h_right = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    h_bias = _on_bf16_grid(values=rng.standard_normal(features).astype(np.float32))
    d_left, d_right, d_bias = upload_activations(h_left), upload_activations(h_right), upload_bias(h_bias)
    try:
        started = time.time()
        d_sum = add_resident(d_left, d_right)
        elapsed = time.time() - started
        d_biased = add_bias_resident(d_left, d_bias)
        try:
            actual, biased = download_activations(d_sum), download_activations(d_biased)
        finally:
            d_sum.free()
            d_biased.free()
    finally:
        d_left.free()
        d_right.free()
        d_bias.free()

    for name, device, expected in (("sum", actual, h_left + h_right), ("bias sum", biased, h_left + h_bias)):
        expected = _on_bf16_grid(values=expected)
        assert np.array_equal(device, expected), (
            f"device {name} differs from the host by up to {np.abs(device - expected).max():.3e}")
    return f"[{tokens},{features}] + [{tokens},{features}] in {elapsed * 1e3:.0f} ms, and + bias row, exact"


def check_prefill_attention(*, tokens: int = 20, heads: int = 12, model: int = 768) -> str:
    """Runs a causal prefill shorter than a VREG for every head on the device.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the device disagrees with the host model.
    """
    rng = np.random.default_rng(23)
    shape = (tokens, model)
    query = _on_bf16_grid(values=rng.standard_normal(shape).astype(np.float32))
    key = _on_bf16_grid(values=rng.standard_normal(shape).astype(np.float32))
    value = _on_bf16_grid(values=rng.standard_normal(shape).astype(np.float32))

    started = time.time()
    actual = prefill_attention(query=query, key=key, value=value, heads=heads, causal=True)
    elapsed = time.time() - started

    head = model // heads
    expected = np.empty(shape, dtype=np.float32)
    mask = np.arange(tokens)[None, :] > np.arange(tokens)[:, None]
    for h in range(heads):
        span = slice(h * head, (h + 1) * head)
        scores = (query[:, span].astype(np.float64) @ key[:, span].astype(np.float64).T)
        scores = np.where(mask, -np.inf, scores / math.sqrt(head))
        scores = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        expected[:, span] = (weights / weights.sum(axis=1, keepdims=True)) @ value[:, span]

    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 0.05, f"prefill attention relative error {error:.4%} is too large"
    return (f"{heads} heads x {tokens} causal queries, 3 launches in {elapsed:.1f} s, "
            f"relative error {error:.2%}")


def check_decode_attention(*, heads: int = 12, head: int = 64, context: int = 48) -> str:
    """Runs one decode step's attention for every head on the device.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the device disagrees with the host model.
    """
    rng = np.random.default_rng(19)
    query = _on_bf16_grid(values=rng.standard_normal((heads, head)).astype(np.float32))
    keys = _on_bf16_grid(values=rng.standard_normal((heads, context, head)).astype(np.float32))
    values = _on_bf16_grid(values=rng.standard_normal((heads, context, head)).astype(np.float32))

    started = time.time()
    actual = decode_attention(query=query, keys=keys, values=values)
    elapsed = time.time() - started

    expected = np.empty((heads, head), dtype=np.float32)
    for h in range(heads):
        scores = (query[h].astype(np.float64) @ keys[h].astype(np.float64).T) / math.sqrt(head)
        scores = scores - scores.max()
        weights = np.exp(scores)
        expected[h] = (weights / weights.sum()) @ values[h].astype(np.float64)

    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 0.05, f"decode attention relative error {error:.4%} is too large"
    return (f"{heads} heads x 1 query over {context} keys, 3 launches in {elapsed:.1f} s, "
            f"relative error {error:.2%}")


def check_layernorm(*, rows: int = 8) -> str:
    """Runs LayerNorm over several hidden rows and verifies it against the host reference."""
    rng = np.random.default_rng(0)
    values = _on_bf16_grid(values=(rng.standard_normal((rows, _MODEL_WIDTH)) * 2.0).astype(np.float32))
    gamma = _on_bf16_grid(values=(1.0 + 0.1 * rng.standard_normal(_MODEL_WIDTH)).astype(np.float32))
    beta = _on_bf16_grid(values=(0.1 * rng.standard_normal(_MODEL_WIDTH)).astype(np.float32))
    actual = layernorm_rows(rows=values, gamma=gamma, beta=beta)
    expected = layernorm_reference(rows=values, gamma=gamma, beta=beta)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _SOFTMAX_ERROR_LIMIT:
        message = (
            f"Unable to verify LayerNorm against the host reference. The relative error must stay "
            f"below {_SOFTMAX_ERROR_LIMIT:.0%}, but got {error:.3%}."
        )
        raise AssertionError(message)
    return f"[{rows},{_MODEL_WIDTH}] rows, relative error {error:.2%}"


def check_transformer_block(*, tokens: int = 64) -> str:
    """Runs one GPT-2 transformer block on the device and verifies it against the host reference.

    Notes:
        The whole block runs resident on the device: two LayerNorms, all four projections, every bias and
        residual add, and twelve attention heads.

    Args:
        tokens: Sequence length, which the softmax fixes at one VREG's width.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the relative error exceeds what a block's chained BF16 steps explain.
    """
    rng = np.random.default_rng(1)
    hidden = _on_bf16_grid(values=(rng.standard_normal((tokens, _MODEL_WIDTH)) * 0.5).astype(np.float32))
    weights = _block_weights(model=_MODEL_WIDTH, seed=2)

    started = time.perf_counter()
    actual = transformer_block(hidden=hidden, weights=weights)
    elapsed = time.perf_counter() - started
    expected = transformer_block_reference(hidden=hidden, weights=weights)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _BLOCK_ERROR_LIMIT:
        message = (
            f"Unable to verify one transformer block against the host reference. The relative error "
            f"must stay below {_BLOCK_ERROR_LIMIT:.0%}, which is what two LayerNorms, thirteen "
            f"products and twelve softmaxes contribute in BF16, but got {error:.3%}."
        )
        raise AssertionError(message)
    return (f"[{tokens},{_MODEL_WIDTH}] block, 12 heads, in {elapsed:.1f} s, "
            f"relative error {error:.2%}")


def _image_of(kernel) -> bytes:
    """Serialises a kernel, without caching.

    Notes:
        Only the launch-path checks reach this, and they run once. Every hot path goes through
        `_kernel_and_image`, which keys on the arguments that built the kernel rather than on the object.

    Args:
        kernel: An assembled kernel.

    Returns:
        Its padded byte image.
    """
    return bytes(kernel.image.to_bytes(pad_to=4))


def _kernel_and_image(key: tuple, build) -> tuple:
    """Returns an assembled kernel and its serialised image, building each once.

    Args:
        key: The build arguments, which identify the kernel.
        build: Builds it when the key is new.

    Returns:
        The (kernel, image bytes) pair.
    """
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        kernel = build()
        cached = (kernel, bytes(kernel.image.to_bytes(pad_to=4)))
        _KERNEL_CACHE[key] = cached
    return cached


def _compiled_gemm(*, depth: int, columns: int, activate: bool) -> bytes:
    """Returns the compiled image that runs one GEMM shape with the contraction summed on the device.

    Args:
        depth: Contraction depth the product covers.
        columns: Output columns the product covers.
        activate: Determines whether the left operand passes through the GELU before it enters the array.

    Returns:
        The image bytes, padded to the loader's boundary.

    Raises:
        ValueError: If no compiled image covers the shape, or if the image drains other than one partial per
            column tile for a block of token rows.
    """
    key = (depth, columns, activate)
    cached = _COMPILED_GEMMS.get(key)
    if cached is not None:
        return cached

    manifest = json.loads((_IMAGE_DIRECTORY / "manifest.json").read_text())
    found = next((name for name, entry in manifest.items()
                  if (entry["depth"], entry["columns"], entry["activate"]) == key), None)
    if found is None:
        message = (
            f"Unable to launch a [{depth}, {columns}] GEMM{' activating its left operand' if activate else ''}. "
            f"No image in {_IMAGE_DIRECTORY} covers that shape, so add it to tools/build_gpt2_images.py and rerun "
            f"that tool on a host."
        )
        raise ValueError(message)

    # Holds the image to the drain these drivers allocate, since an image draining any other layout writes its
    # product into the wrong rows rather than failing.
    expected = _ZERO_TILE_WORDS + columns // _TILE * _BLOCK_TOKENS * _TILE
    entry = manifest[found]
    if entry["drain_words"] != expected:
        message = (
            f"Unable to launch the image {found}. A product drains {expected} words, one partial per column tile, "
            f"but the image drains {entry['drain_words']}."
        )
        raise ValueError(message)

    image_bytes = (_IMAGE_DIRECTORY / found).read_bytes()
    _COMPILED_GEMMS[key] = image_bytes
    return image_bytes


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
    minitpuSetDevice(single_halt.uio_name)
    tensor = DeviceTensor(base=0, tokens=tokens, features=features)
    tensor.base = minitpuMalloc(tensor.nbytes)
    host = minitpuHostAlloc(_ZERO_TILE_BYTES)
    host.view.view("<u2")[:_ZERO_TILE_WORDS] = 0
    for block in range(tensor.blocks):
        minitpuMemcpy(tensor.d_drain(block), host, _ZERO_TILE_BYTES, MemcpyKind.HostToDevice)
    return tensor


def _sequence_length(*, rows: int, valid: int, sequences: int) -> int:
    """Rows one sequence owns, for the mask that keeps sequences apart.

    Notes:
        One sequence needs no boundary: the padded tail is excluded by column and the causal rule excludes
        everything after a query, so the sequence is the whole padded axis and a prompt of any length works.
        A batch does need the boundary, and needs it to tile the padded axis, the mask placing each sequence
        by index.

    Args:
        rows: The padded token axis.
        valid: Real rows in it.
        sequences: Independent sequences laid end to end.

    Returns:
        The rows between one sequence's start and the next.

    Raises:
        ValueError: If a batch's sequences do not tile the padded axis.
    """
    if sequences == 1:
        return rows
    if valid % sequences or rows % (valid // sequences):
        message = (
            f"Unable to split {valid} real rows of {rows} into {sequences} "
            f"sequences. A batch lays equal-length sequences end to end, and the "
            f"padded axis must be a whole number of them."
        )
        raise ValueError(message)
    return valid // sequences


def _attention_mask(rows: int, heads: int, *, causal: bool, sequence: int,
                    valid: int, span: int) -> DeviceTensor:
    """Returns the additive mask for a shape, uploading it once.

    Notes:
        Three things are forbidden and all three are the same constant: a key past the end of the array, a key
        in another sequence, and when causal a key after the query's own position. The second is why the batch
        axis needs no kernel of its own. The constant is large and negative rather than -inf, the kernel
        subtracting the row maximum before the exponential and inf - inf being a NaN.

    Args:
        rows: The padded token axis.
        heads: Attention heads.
        causal: Whether a query may only attend at or before its own position.
        sequence: Rows one sequence owns.
        valid: Real rows; the rest of the axis is padding no query may attend to.
        span: Keys a score row covers.

    Returns:
        The [rows, heads * span] mask, resident.
    """
    key = (rows, heads, causal, sequence, valid, span)
    d_cached = _ATTENTION_MASKS.get(key)
    if d_cached is not None:
        return d_cached

    index = np.arange(rows)
    forbidden = index[:, None] // sequence != index[None, :] // sequence
    forbidden |= index[None, :] >= valid
    if causal:
        forbidden |= index[None, :] % sequence > index[:, None] % sequence
    # A row may always attend to itself, which every real query does already and the padded rows need: a row
    # forbidding everything leaves the softmax subtracting -1e30 from itself. Their output is discarded, so
    # what matters is only that it is a number.
    forbidden &= index[:, None] != index[None, :]

    h_mask = np.zeros((rows, heads * span), dtype=np.float32)
    for head in range(heads):
        h_mask[:, head * span:(head + 1) * span][:, :rows][forbidden] = -1.0e30
        h_mask[:, head * span:(head + 1) * span][:, rows:] = -1.0e30
    d_cached = upload_activations(h_mask)
    _ATTENTION_MASKS[key] = d_cached
    return d_cached


def _grouped_operands(rows: int, heads: int, head: int, span: int):
    """Allocates the per-head K-transpose and V operands for a grouped attention.

    Notes:
        These replace the block-diagonal, which encodes independence by surrounding each head's operand with
        zeros and costs a factor of `heads` in multiplies, where laying the same operands end to end costs a
        loop level. They are `[heads, head_dim, span]` for the transposed keys and `[heads, span, head]` for
        the values, which is what `build_looped_gemm(batch=heads)` walks.

    Args:
        rows: Padded rows attended over.
        heads: Attention heads.
        head: The head dimension.
        span: Keys a score row covers, rows rounded up to whole VREGs.

    Returns:
        The key operand and the value operand, as `ResidentWeights` over the whole
        stack, the GEMM being given the first problem's address and walking the rest.
    """
    key = (rows, heads, head, span)
    if key in _GROUPED_OPERANDS:
        return _GROUPED_OPERANDS[key]

    # `depth` and `columns` describe ONE problem; the launch walks `heads` of them.
    d_keys = ResidentWeights(minitpuMalloc(heads * head * span * _ELEMENT_BYTES),
                             head, span, heads * head * span * _ELEMENT_BYTES)
    d_values = ResidentWeights(minitpuMalloc(heads * span * head * _ELEMENT_BYTES),
                               span, head, heads * span * head * _ELEMENT_BYTES)
    for d_operand in (d_keys, d_values):
        h_zero = minitpuHostAlloc(d_operand.nbytes)
        h_zero.view[:d_operand.nbytes] = 0
        minitpuMemcpy(d_operand.address, h_zero, d_operand.nbytes,
                      MemcpyKind.HostToDevice)

    _GROUPED_OPERANDS[key] = (d_keys, d_values)
    return _GROUPED_OPERANDS[key]


def _grouped_product(d_left, d_weights: ResidentWeights, *, heads: int, rows: int,
                     columns: int, name: str) -> DeviceTensor:
    """Runs `heads` independent products in one launch per token block.

    Args:
        d_left: The stacked left operands; problem h's is a contiguous slice.
        d_weights: The stacked right operands, `heads` of them end to end.
        heads: Problems in the launch.
        rows: Padded token rows.
        columns: Output columns of ONE problem.
        name: Cache key prefix for the assembled image.

    Returns:
        The [rows, heads * columns] product, resident and packed.
    """
    depth = d_weights.depth
    kernel, image_bytes = _kernel_and_image(
        (name, columns // _TILE, depth, heads),
        lambda: build_looped_gemm(
            column_tiles=columns // _TILE, groups=1, activate=False,
            loop_blocks=depth // _BLOCK_DEPTH, batch=heads,
            batch_activation_bytes=_BLOCK_TOKENS * depth * _ELEMENT_BYTES))
    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)
    d_out = _allocate_tensor(rows, heads * columns)
    for block in range(rows // _BLOCK_TOKENS):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_left.operand(block), d_weights.address, d_out.d_drain(block)])
    return d_out


def _pad_rows_to_tile(h_rows: NDArray[np.float32]) -> NDArray[np.float32]:
    """Zero-extends the row axis to a whole number of transpose tiles.

    Notes:
        The staging tile is square, so a transpose needs both dimensions to be a whole number of `_TILE`.
        Padding with zeros is exact, the caller dropping the padded columns before they reach a score.
    """
    rows = h_rows.shape[0]
    remainder = rows % _TILE
    if remainder == 0:
        return h_rows
    extended = np.zeros((rows + _TILE - remainder, h_rows.shape[1]), dtype=np.float32)
    extended[:rows] = h_rows
    return extended


def _pad_tokens_for_launch(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    """Zero-extends the token axis to the whole number of rows one launch covers."""
    tokens = rows.shape[0]
    remainder = tokens % _BLOCK_TOKENS
    if remainder == 0:
        return rows
    extended = np.zeros((tokens + _BLOCK_TOKENS - remainder, rows.shape[1]), dtype=np.float32)
    extended[:tokens] = rows
    return extended


def _block_weights(model: int, seed: int) -> dict:
    """Builds one block's parameters at GPT-2 initialisation scales, on the BF16 grid."""
    rng = np.random.default_rng(seed)
    hidden = 4 * model

    def normal(shape: tuple, scale: float) -> NDArray[np.float32]:
        return _on_bf16_grid(values=(rng.standard_normal(shape) * scale).astype(np.float32))

    return {
        "ln_1.g": _on_bf16_grid(values=np.ones(model, dtype=np.float32)),
        "ln_1.b": normal((model,), 0.02),
        "attn.qkv.w": normal((model, 3 * model), 0.02),
        "attn.qkv.b": normal((3 * model,), 0.02),
        "attn.proj.w": normal((model, model), 0.02),
        "attn.proj.b": normal((model,), 0.02),
        "ln_2.g": _on_bf16_grid(values=np.ones(model, dtype=np.float32)),
        "ln_2.b": normal((model,), 0.02),
        "mlp.fc.w": normal((model, hidden), 0.02),
        "mlp.fc.b": normal((hidden,), 0.02),
        "mlp.proj.w": normal((hidden, model), 0.02),
        "mlp.proj.b": normal((model,), 0.02),
    }


def main() -> int:
    """Runs the demo programs on the board and reports each result against a host model.

    Returns:
        Zero when every program matched its expected output, and one when any of them did not.
    """
    outcomes = [
        run_check(name="Halt", check=check_halt),
        run_check(name="COPY round trip", check=check_copy_argument),
        run_check(name="Full GEMM", check=check_full_gemm),
        run_check(name="Elementwise add on device", check=check_elementwise_add),
        run_check(name="Prefill attention, all heads", check=check_prefill_attention),
        run_check(name="Decode attention, all heads", check=check_decode_attention),
        run_check(name="LayerNorm rows", check=check_layernorm),
        run_check(name="Transformer block", check=check_transformer_block),
    ]
    print(f"\n{sum(outcomes)} of {len(outcomes)} checks passed")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
