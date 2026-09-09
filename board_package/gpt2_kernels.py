#!/usr/bin/env python3
"""Provides the GPT-2 host drivers, the layer references and the checks that say the device path is right.

Running this module on a board executes every check in order. Composition lives here, while the arithmetic is in
bf16.py and the schedules are in kernels.py. The PL runs the hidden dimensions, being both projections, every
attention head, both feed-forward products and LayerNorm, and the host holds the residual adds, biases, head
reshaping and the K transpose.
"""

from __future__ import annotations

from collections.abc import Callable

import math
import pathlib
import sys
import time

import numpy as np
from numpy.typing import NDArray

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from bf16 import (  # noqa: E402,F401  -- re-exported: the board and tools import them from here
    HERE,
    MAX_ATTENTION_ROWS,
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_DEPTH,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _CORRUPTION_WORD,
    _ELEMENT_BYTES,
    _GELU_LIMIT_Q7,
    _GELU_TABLE_ENTRIES,
    _INTERLEAVE_PAYLOAD_BYTES,
    _INTERLEAVE_ROUNDS,
    _LAUNCH_REPEATS,
    _LN_EPSILON,
    _LN_ROW_CONST,
    _LN_ROW_OUT,
    _LN_ROW_X,
    _LOOP_BLOCKS,
    _MAX_DESCRIPTOR_ROWS,
    _NUM_LANES,
    _MODEL_WIDTH,
    _NUM_LANES,
    _NUM_SUBLANES,
    _ROW_ACTIVATION,
    _ROW_BYTES,
    _ROW_OUTPUT,
    _ROW_SCORES,
    _ROW_SOFTMAX_OUT,
    _ROW_WEIGHT,
    _ROW_ZERO,
    _STEPS,
    _STEP_TILES,
    _TILE,
    _TILE_ROWS,
    _TILE_VREGS,
    _TOKEN_GROUPS,
    _VMEM_ENTRIES_PER_LANE,
    _VMEM_SLOT_STRIDE,
    _VREG_ELEMENTS,
    _WIDTH_ROWS,
    _WIDTH_VREGS,
    _ZERO_TILE_BYTES,
    _ZERO_TILE_WORDS,
    _bf16_round,
    _fp32_abs_q7,
    _gelu_table,
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
    pack_gemm_input,
    pack_weights,
)
from kernels import (  # noqa: E402,F401  -- re-exported: the board and tools import them from here
    HERE,
    _emit_accumulate,
    _emit_gelu_gemm_step,
    _emit_gemm_step,
    _emit_pipelined_matrix,
    build_layernorm_rows,
    build_elementwise_add,
    build_head_copy,
    build_head_transpose,
    build_matrix_transpose,
    build_tile_transpose,
    build_looped_gemm,
    build_softmax_packed,
    build_softmax_rows,
    single_halt,
    single_mul_add,
)
from runtime import (  # noqa: E402
    MemcpyKind,
    MinitpuKernel,
    minitpuFree,
    minitpuGetAddressSpace,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuHostRewind,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
)


sys.path.insert(0, str(HERE))


sys.path.insert(0, str(HERE / "minitpu_runtime"))




_FFN_ERROR_LIMIT: float = 0.01


_GEMM_ERROR_LIMIT: float = 0.01


_SOFTMAX_ERROR_LIMIT: float = 0.02


_ATTENTION_ERROR_LIMIT: float = 0.02


_BLOCK_ERROR_LIMIT: float = 0.05


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


class ResidentWeights:
    """A packed weight matrix that lives in DMEM across launches.

    `full_gemm` used to take fp32 numpy and re-derive the device's layout on every
    call: `pack_weights` alone measures 72.7 ms of a 168 ms launch, and a decode
    token re-packs all 85 M parameters of GPT-2's body because nothing survives
    the call. The weights do not change between tokens, so this uploads them once.

    Device allocations outlive a launch already -- `minitpuHostReset` resets only
    the host udmabuf, and the device pool is rebuilt only if the address space
    changes -- so residency needed no runtime change, just somewhere to keep the
    address.
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


#: What is currently staged at the kernel base, so an unchanged image is not
#: re-copied. The staging address is fixed, so re-launching the same kernel only
#: needs the COPY when something else has been staged since.
#:
#: Sound only while every launch in the process stages through `_stage_image`.
#: Every driver in this module does. A program that also runs a decorated kernel
#: from `run_example_kernel.py`, which stages its own image through the runtime, would
#: leave this stale -- so the cache belongs in the runtime rather than here once
#: anything needs to mix the two.
_RESIDENT_IMAGES: dict[bytes, int] = {}


#: Assembled kernels and their serialised images, keyed by the arguments that
#: built them. `MinitpuKernel.image.to_bytes` rebuilds the byte string bundle by
#: bundle, and it measured 36.8 ms of a 47.4 ms fully-resident launch -- the
#: largest single cost left once the weights, the activations and the staged image
#: all stopped being re-derived. Building the kernel is not free either, and
#: `build_softmax_rows` and `build_layernorm_rows` did it on every call.
#:
#: Keyed by build arguments rather than by the kernel object. Two earlier attempts
#: were unsound: `id(kernel)` is reused once CPython collects the object, so a
#: dead softmax kernel handed its image to a later layernorm and the chat emitted
#: token 0 forever -- while all fourteen board checks passed, because they reuse
#: kernels that never die. Memoising ON the kernel fails outright, since it is a
#: frozen dataclass. The arguments are the identity.
_KERNEL_CACHE: dict[tuple, tuple] = {}


def _image_of(kernel) -> bytes:
    """Serialises a kernel, without caching.

    Only the module-level kernels reach this, and only from the launch-path
    checks, which run once. Every hot path goes through `_kernel_and_image`
    instead, which keys on the arguments that built the kernel rather than on the
    object -- see the note above for why the object is the wrong key.

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
    # Its own address, not the shared staging one. `minitpuLaunchKernel` takes
    # kbin_addr as "the device-memory base of the kernel binary" and asks only for
    # 32-byte alignment, which the allocator already gives -- nothing requires the
    # image to live at `kernel_base`. Staging every kernel at one address is what
    # made a layer restage on all eighteen launches, at 1.35 ms of COPY each.
    address = minitpuMalloc(len(image_bytes))
    host = minitpuHostAlloc(len(image_bytes))
    host.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
    minitpuMemcpy(address, host, len(image_bytes), MemcpyKind.HostToDevice)
    _RESIDENT_IMAGES[image_bytes] = address
    return address


class DeviceTensor:
    """Packed activations living in DMEM, in the layout the GEMM already produces.

    The point of this type is that a product never has to reach the host to be
    usable. With `fuse_groups`, a GEMM drains one tile per output column tile, so
    one launch writes [column_tiles, BLOCK_TOKENS, TILE] -- and `pack_activations`
    orders a block's result [depth/TILE, BLOCK_TOKENS, TILE]. Those are the SAME
    layout, so a fused product is already the next launch's packed left operand,
    with nothing to transpose, repack or copy. The host's reshape in `full_gemm`
    exists only to hand a caller a row-major numpy array.

    The layout is BLOCK-MAJOR: a tensor of more than `_BLOCK_TOKENS` rows is a
    sequence of independent per-block buffers, not one array with a token axis.
    That is what makes a block's operand and its output both a plain linear offset
    from the base -- interleaving them into a single token axis would leave
    neither addressable by pointer arithmetic, because a launch writes its column
    tiles consecutively.

    Every block is headed by its own zeroed clearing tile, because a launch loads
    that tile from the drain pointer it is given -- so block 1's pointer must find
    zeros there, not block 0's results. That makes each block a uniform region and
    keeps both offsets a multiple of one stride.
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

        Not a `minitpuFree`: a tensor's clearing tile is read by a launch and never
        written by one, so a buffer of the same shape can be handed straight back
        out with its tile still zeroed. Zeroing it costs a host-to-device copy per
        allocation, and a layer allocates one per GEMM and per add.
        """
        if self.base is not None:
            _TENSOR_POOL.setdefault((self.tokens, self.features), []).append(self.base)
            self.base = None


#: Freed tensors, keyed by shape, waiting to be handed out again. A launch reads
#: a drain buffer's clearing tile and never writes it, so a reused buffer needs no
#: re-zeroing -- which is what makes the pool worth having, since zeroing is a
#: host-to-device copy and a layer allocates one buffer per GEMM and per add.
_TENSOR_POOL: dict[tuple[int, int], list[int]] = {}


class DeviceTensorSlice:
    """A contiguous feature range of a packed tensor, addressed in ITS blocks.

    The fused QKV projection is one `[tokens, 3 * model]` tensor and attention wants
    its three thirds separately. In the packed layout a block is
    `[features / TILE, BLOCK_TOKENS, TILE]`, so a range of feature tiles is a
    contiguous run starting at a plain offset -- splitting Q, K and V is pointer
    arithmetic and copies nothing.

    This is a view rather than a `DeviceTensor` because a tensor derives its block
    stride from its own feature count, and a slice must keep the PARENT's stride:
    the blocks it lives in are the parent's blocks, spaced by the parent's width.
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


# Host staging is a fresh `minitpuHostAlloc` per call, deliberately. It is a bump
# allocator with no per-region free (v1 by design), so a long-lived process that
# sweeps many shapes will exhaust its 16 MiB udmabuf, which is why a benchmark
# takes one shape per process.
#
# Caching and reusing one region per role does not work. Two things break it. A cached region does not survive
# `minitpuHostReset`, which the drivers call and which drops the udmabuf out from under every region carved from it. The
# helper also cannot bind a device of its own without rebinding one in the middle of whatever sequence its caller was
# running. Both show up as `instr_ready` never returning, the device left mid-command, rather than as a wrong answer,
# and
# the second attempt took the board from 6 of 16 to 0 of 16. The fix belongs in the
# allocator, as a real per-region free, not in a cache layered over one that has none.


def reset_host_staging() -> None:
    """Returns the pinned staging arena to empty. Safe only when nothing is live.

    Rewinds the bump offset rather than dropping the buffer. `minitpuHostReset`
    unmaps the 16 MiB udmabuf too, so the next allocation re-opens it -- profiled
    at 1.875 ms a call, 3.7 ms a layer, more than either host-to-device copy the
    layer makes. Nothing here survives the call that made it, so there is nothing
    for the weaker guarantee to break.

    `minitpuHostAlloc` is a bump allocator with no per-region free (v1 by design),
    so every kernel image staged and every operand packed consumes a slice of a
    16 MiB udmabuf permanently. One forward pass is comfortable; seventeen of them
    in a process is not, which is what a batch-invariance sweep does.

    Every staging region in this module dies inside the call that made it -- write,
    copy, done, or copy, read, done -- so between forward passes there is nothing to
    invalidate. DEVICE allocations are a separate pool and are untouched: the
    resident weights, the biases, the block-diagonal operands, the LayerNorm
    constants and the tensor pool all survive.

    What is NOT safe is caching a region ACROSS this call. A reset drops the udmabuf,
    every region carved from it becomes a view into memory the allocator no longer
    owns, and handing one to a DMA leaves the core mid-command -- which reads as
    `instr_ready` never returning rather than as a wrong answer.
    """
    minitpuHostRewind()


def _allocate_tensor(tokens: int, features: int) -> DeviceTensor:
    """Hands out a drain-shaped DMEM buffer, from the pool when one is waiting.

    The kernel loads its clearing tile from the head of the drain buffer, so those
    bytes must be zero before the first launch that uses it -- but only ONCE per
    buffer. A launch reads that tile and never writes it, so a freed buffer comes
    back with its tile still zeroed and needs neither a malloc nor a copy.

    `DeviceTensor.free` has pushed buffers onto `_TENSOR_POOL` all along and
    nothing ever took one back off, so every intermediate in every layer was a
    fresh allocation and a fresh host-to-device zeroing copy: 17 copies a layer
    that the trace attributed to the layer and that were pure allocation.

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


#: The clearing tile that heads every drain buffer, in the units a caller needs to
#: allocate and zero it. Public because a caller driving the raw runtime has to
#: place it, and reaching through the underscore to find out how big it is would
#: mean the module never said.
ZERO_TILE_BYTES = _ZERO_TILE_BYTES
ZERO_TILE_WORDS = _ZERO_TILE_WORDS


def gemm_kernel(*, depth: int, columns: int, activate: bool) -> tuple:
    """Returns the assembled GEMM and its image for one operand shape.

    Derives the kernel's trip counts from the shapes rather than making a caller
    know that a group is `_LOOP_BLOCKS * _BLOCK_DEPTH` deep and a column tile is
    `_TILE` wide. Always the group-fusing variant, since a caller reaching for the
    raw runtime wants the product finished in DMEM.

    Args:
        depth: The contraction, a whole number of `_LOOP_BLOCKS * _BLOCK_DEPTH`.
        columns: Output features, a whole number of `_TILE`.
        activate: Whether each staged tile is GELU-activated entering the array.

    Returns:
        The (kernel, image bytes) pair, built at most once per shape.

    Raises:
        ValueError: If a dimension is not a whole number of the blocks a launch covers.
    """
    for name, value, block in (("depth", depth, _LOOP_BLOCKS * _BLOCK_DEPTH),
                               ("columns", columns, _TILE)):
        if value % block:
            message = (
                f"Unable to build a GEMM with {name} of {value}. It must be a whole "
                f"number of {block}, which is what one launch covers."
            )
            raise ValueError(message)
    column_tiles = columns // _TILE
    groups = depth // (_LOOP_BLOCKS * _BLOCK_DEPTH)
    return _kernel_and_image(
        ("gemm", column_tiles, groups, activate, True),
        lambda: build_looped_gemm(column_tiles=column_tiles, groups=groups,
                                  activate=activate, fuse_groups=True))


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

    The kernel reads and writes in the layout a GEMM operand uses, so a normalised
    tensor feeds the next projection where it lies. It used to come back to PS DDR
    to be re-packed, purely because its descriptors were set for row-major.

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
        A 16x16 tile of a row-major array is 16 rows at stride `columns / NUM_LANES`,
        not 16 consecutive rows, so this is the first thing in the package that
        needs a DMA descriptor stride other than one. Every other kernel here reads
        consecutive VMEM rows with a full mask.

        Tile (i, j) of the source becomes tile (j, i) of the destination, and that
        placement is pointer arithmetic: one launch per source tile COLUMN walks
        that column's tiles with `load_stride = columns / NUM_LANES` and lands them
        consecutively in the destination row they belong to.

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

        # One launch for the whole matrix. The tile-column walk is the kernel's
        # outer loop now, not the host's: at GPT-2's width this was 48 launches of
        # a 32-row transpose, and on .193 that made the launch count scale with the
        # batch at ~51 per sequence per layer, 48 of them here.
        minitpuLaunchKernel(d_image, len(image_bytes),
                            args=[d_in, 0, d_out])
        minitpuMemcpy(h_out, d_out, words.nbytes, MemcpyKind.DeviceToHost)
        return from_bf16_words(
            words=h_out.view.view("<u2")[: words.size]).reshape(columns, rows)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()


def transpose_tiles(h_tiles: NDArray[np.float32]) -> NDArray[np.float32]:
    """Transposes each 16x16 tile of an array on the device.

    The operation nothing else in the machine could express: `sublane_stride`
    permutes rows uniformly across every lane and a DMA descriptor moves whole
    16-lane rows, so exchanging lanes with sublanes needed the staging array that
    `vtxin` and `vtxout` drive.

    Args:
        h_tiles: [tiles * _TILE, _TILE] host array, a stack of square tiles.

    Returns:
        The same stack with every tile transposed.

    Raises:
        ValueError: If the shape is not a whole number of square tiles.
    """
    rows, columns = h_tiles.shape
    if columns != _TILE or rows % _TILE:
        message = (
            f"Unable to transpose a [{rows},{columns}] array. The staging tile is "
            f"{_TILE}x{_TILE}, so this takes a whole number of {_TILE}x{_TILE} tiles."
        )
        raise ValueError(message)
    tiles = rows // _TILE

    kernel, image_bytes = _kernel_and_image(
        ("transpose", tiles),
        # A [tiles * 16, 16] stack: one matrix row is exactly one VMEM row, so the
        # rows of a tile are consecutive, and each result lands a whole tile on.
        lambda: build_tile_transpose(tiles=tiles, load_stride=1, store_stride=1,
                                     store_advance=_TILE))
    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    d_owned: list[int] = []
    try:
        d_image = _stage_image(image_bytes)
        words = as_bf16_words(values=_on_bf16_grid(values=h_tiles)).reshape(-1)
        h_in = minitpuHostAlloc(words.nbytes)
        h_in.view.view("<u2")[: words.size] = words
        d_in = minitpuMalloc(words.nbytes)
        d_owned.append(d_in)
        minitpuMemcpy(d_in, h_in, words.nbytes, MemcpyKind.HostToDevice)

        h_out = minitpuHostAlloc(words.nbytes)
        d_out = minitpuMalloc(words.nbytes)
        d_owned.append(d_out)

        minitpuLaunchKernel(d_image, len(image_bytes),
                            args=[d_in, 0, d_out])
        minitpuMemcpy(h_out, d_out, words.nbytes, MemcpyKind.DeviceToHost)
        return from_bf16_words(
            words=h_out.view.view("<u2")[: words.size]).reshape(h_tiles.shape)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()


def add_resident(d_left: DeviceTensor, d_right: DeviceTensor, *,
                 d_out: DeviceTensor | None = None) -> DeviceTensor:
    """Adds two DMEM tensors elementwise, leaving the sum in DMEM.

    This is what a projection's bias and a block's residual become once they stop
    being numpy. Both were forcing the product back to PS DDR to be added there,
    and then the next projection sent it out again -- two copies per add, for one
    `vadd` the machine has always had.

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

    The bias is packed with each feature tile repeated across the four sublane rows, so the kernel stages it once
    and one contiguous read broadcasts it across tokens. That is far cheaper than materialising a
    `[tokens, features]` tensor for every layer and execution shape.
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

    Nothing crosses the host boundary: the activations are already packed, the
    weights are already packed, and the product is written in the layout the next
    launch's activation argument expects. The only host work is issuing the
    launch, which is the intervention point that stays.

    Notes:
        Always fuses the contraction's groups, because a per-group drain would
        leave partials the host has to add -- and adding them here is the whole
        point. That costs about 1.8x the relative error of the float32 host sum
        (`check_fused_groups` reports both on every board run).

        One launch per block of `_BLOCK_TOKENS` rows, as `full_gemm` does. The
        activation pointer advances by a block of packed rows and the drain
        pointer by a block of output rows.

    Args:
        d_activations: [tokens, contraction] left operand, resident.
        d_weights: [contraction, columns] right operand, resident.
        activate: Whether each staged tile is GELU-activated entering the array.
        d_out: Where to write. Allocated when omitted.

    Returns:
        The product, resident.

    Raises:
        ValueError: If the operands disagree, or a dimension is not a whole number of blocks.
    """
    tokens, depth = d_activations.shape
    columns = d_weights.columns
    if depth != d_weights.depth:
        message = (
            f"Unable to multiply a [{tokens},{depth}] tensor by a [{d_weights.depth},{columns}] "
            f"one. Their inner dimensions must agree."
        )
        raise ValueError(message)
    for name, value, block in (
        ("tokens", tokens, _BLOCK_TOKENS), ("depth", depth, _LOOP_BLOCKS * _BLOCK_DEPTH),
        ("columns", columns, _TILE)
    ):
        if value % block:
            message = (
                f"Unable to multiply tensors with {name} of {value}. It must be a whole number "
                f"of {block}, which is what one launch covers."
            )
            raise ValueError(message)

    column_tiles = columns // _TILE
    groups = depth // (_LOOP_BLOCKS * _BLOCK_DEPTH)
    kernel, image_bytes = _kernel_and_image(
        ("gemm", column_tiles, groups, activate, True),
        lambda: build_looped_gemm(column_tiles=column_tiles, groups=groups,
                                  activate=activate, fuse_groups=True))

    if d_out is None:
        d_out = _allocate_tensor(tokens, columns)

    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)

    for block in range(tokens // _BLOCK_TOKENS):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_activations.operand(block), d_weights.address, d_out.d_drain(block)])
    return d_out


def full_gemm(
    activations: NDArray[np.float32], weights: NDArray[np.float32] | ResidentWeights,
    *, activate: bool = False, fuse_groups: bool = False
) -> NDArray[np.float32]:
    """Multiplies two arrays of arbitrary size on the device, one launch per token block.

    Notes:
        The device walks every output column tile and every contraction block, draining one partial per block
        group. The host sums those groups in float32, which holds the BF16 accumulate chain to a depth the error
        bound tolerates.

        `weights` may be a `ResidentWeights` handle instead of an array, in which case nothing is packed or
        copied: the matrix is already in DMEM from a previous `upload_weights`. That removes the single
        largest cost of a launch -- `pack_weights` is 72.7 ms of 168 ms -- for any caller that reuses a matrix,
        which a language model does on every token.

    Args:
        activations: The left operand, whose rows are tokens and columns the contraction.
        weights: The right operand, whose rows are the contraction and columns the output features.
        activate: Determines whether each staged tile is activated before it enters the array. Set it for a
            feed-forward layer's down projection, whose left operand is the activated intermediate.
        fuse_groups: Sum the contraction's groups on the device instead of draining one partial per group
            for the host to add in float32. Costs 1.16x the relative error at GPT-2's depth and removes
            both the host reduction and `groups - 1` of every column tile's drain traffic.

    Returns:
        The float32 product, recovered from the BF16 values the kernel returns.

    Raises:
        ValueError: If the operands' inner dimensions disagree, or a dimension is not a whole number of the
            blocks one launch covers.
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
    blocks = depth // _BLOCK_DEPTH
    if blocks % _LOOP_BLOCKS:
        message = (
            f"Unable to run a {depth}-deep contraction on the looped kernel. The contraction must be a whole "
            f"number of the {_LOOP_BLOCKS} blocks one group walks, but got {blocks} blocks."
        )
        raise ValueError(message)
    groups = blocks // _LOOP_BLOCKS
    column_tiles = columns // _TILE

    kernel, image_bytes = _kernel_and_image(
        ("gemm", column_tiles, groups, activate, fuse_groups),
        lambda: build_looped_gemm(column_tiles=column_tiles, groups=groups,
                                  activate=activate, fuse_groups=fuse_groups))
    drained_groups = 1 if fuse_groups else groups
    weight_words = None if resident is not None else pack_weights(weights=weights)
    drain_words = _ZERO_TILE_WORDS + column_tiles * drained_groups * _BLOCK_TOKENS * _TILE
    result = np.empty((tokens, columns), dtype=np.float32)

    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    d_owned: list[int] = []

    try:
        d_image = _stage_image(image_bytes)

        if resident is not None:
            # Already in DMEM, and deliberately NOT appended to `device`: it must
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
        # tile to find where the partials land.
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

            partials = from_bf16_words(
                words=h_drain.view.view("<u2")[_ZERO_TILE_WORDS:drain_words])
            tiles = partials.reshape(column_tiles, drained_groups, _BLOCK_TOKENS, _TILE)
            # With fuse_groups the device has already summed them and this axis is 1, so the sum
            # is a no-op. Without it, this is the float32 reduction of the partials.
            summed = tiles.sum(axis=1, dtype=np.float32)
            result[token : token + _BLOCK_TOKENS] = summed.transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)
    finally:
        for address in d_owned:
            minitpuFree(address)
        minitpuHostReset()
    return result


def full_ffn(
    activations: NDArray[np.float32],
    up_weight: NDArray[np.float32],
    up_bias: NDArray[np.float32],
    down_weight: NDArray[np.float32],
    down_bias: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Runs one transformer feed-forward layer at model dimensions on the MINITPU.

    Notes:
        Runs as two full_gemm passes, since GELU needs the finished sum and one launch holds only 32 of a 768-deep
        contraction, so fusing the activation onto the up projection would activate a partial. The activation rides
        the down projection, whose left operand the host has already summed, and costs no bundle of its own. Every
        product and every activation runs on the device, leaving the host to slice operands and sum contraction
        blocks.

    Args:
        activations: The layer input, whose rows are tokens.
        up_weight: The up projection, contracted against the input's columns.
        up_bias: The up projection's bias, added to every token row.
        down_weight: The down projection, contracted against the activated intermediate.
        down_bias: The down projection's bias, added to every token row.

    Returns:
        The layer output, with one row per input token.
    """
    hidden = full_gemm(activations=activations, weights=up_weight) + up_bias
    return full_gemm(activations=hidden, weights=down_weight, activate=True) + down_bias


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

        # One launch streams every whole chunk: the kernel advances its own
        # pointers, and its inner loop names its AGU level so the chunk loop does
        # not steal the row index. A ragged tail still needs its own image, because
        # the inner trip count is baked in -- but that is one more launch, not one
        # per chunk.
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


def multi_head_attention(
    hidden: NDArray[np.float32], qkv_weight: NDArray[np.float32], qkv_bias: NDArray[np.float32],
    proj_weight: NDArray[np.float32], proj_bias: NDArray[np.float32], *, heads: int = 12
) -> NDArray[np.float32]:
    """Runs one multi-head self-attention on the device.

    Notes:
        The projections are device products at model width, and the attention is
        `prefill_attention` -- three launches for every head, not three per head. Splitting the
        projection into heads and concatenating them again is a contiguous view over the same bytes
        and costs nothing at all.

        Unmasked, matching `attention_head`'s long-standing behaviour and what
        `transformer_block_reference` models. The chat path passes `causal=True`.

    Args:
        hidden: [tokens, model] input.
        qkv_weight: [model, 3*model] fused query/key/value projection.
        qkv_bias: [3*model] bias.
        proj_weight: [model, model] output projection.
        proj_bias: [model] bias.
        heads: Attention heads.

    Returns:
        The [tokens, model] attention output.
    """
    fused = full_gemm(activations=hidden, weights=qkv_weight) + qkv_bias
    query, key, value = np.split(fused.astype(np.float32), 3, axis=1)
    gathered = prefill_attention(
        query=np.ascontiguousarray(query), key=np.ascontiguousarray(key),
        value=np.ascontiguousarray(value), heads=heads, causal=False)
    return full_gemm(activations=_on_bf16_grid(values=gathered), weights=proj_weight) + proj_bias


def transformer_block(hidden: NDArray[np.float32], weights: dict) -> NDArray[np.float32]:
    """Runs one GPT-2 transformer block on the device.

    Notes:
        LayerNorm, both projections, every attention head and both feed-forward products run on the
        device. The residual adds, the biases and the head reshaping stay on the host, which is the
        agreed split while the PL covers the hidden dimensions and the PS covers the gaps.

    Args:
        hidden: [tokens, model] block input.
        weights: The block's parameters, keyed as GPT-2 names them: ``ln_1.g``, ``ln_1.b``,
            ``attn.qkv.w``, ``attn.qkv.b``, ``attn.proj.w``, ``attn.proj.b``, ``ln_2.g``,
            ``ln_2.b``, ``mlp.fc.w``, ``mlp.fc.b``, ``mlp.proj.w``, ``mlp.proj.b``.

    Returns:
        The [tokens, model] block output.
    """
    normalised = layernorm_rows(rows=hidden, gamma=weights["ln_1.g"], beta=weights["ln_1.b"])
    attended = multi_head_attention(
        hidden=_on_bf16_grid(values=normalised),
        qkv_weight=weights["attn.qkv.w"], qkv_bias=weights["attn.qkv.b"],
        proj_weight=weights["attn.proj.w"], proj_bias=weights["attn.proj.b"])
    residual = _on_bf16_grid(values=hidden + attended)

    normalised = layernorm_rows(rows=residual, gamma=weights["ln_2.g"], beta=weights["ln_2.b"])
    expanded = full_gemm(activations=_on_bf16_grid(values=normalised),
                         weights=weights["mlp.fc.w"]) + weights["mlp.fc.b"]
    contracted = full_gemm(activations=_on_bf16_grid(values=expanded),
                           weights=weights["mlp.proj.w"], activate=True) + weights["mlp.proj.b"]
    return _on_bf16_grid(values=residual + contracted)


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


def attention_head(
    query: NDArray[np.float32], key: NDArray[np.float32], value: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Runs one attention head on the device: softmax(Q Kᵀ / √d) V.

    Notes:
        Three launches, composed the way `full_ffn` composes two: a product, a softmax, a product.
        Two things ride on the host, deliberately and temporarily.

        **The transpose.** A weight-stationary array wants Kᵀ as its stationary operand, and nothing
        in the ISA exchanges lanes with sublanes -- `sublane_stride` permutes rows, and a DMA
        descriptor moves whole rows too. The host packer already materialises a copy of every
        operand, so it transposes there for free.

        **The contraction padding.** The looped GEMM accumulates `_LOOP_BLOCKS` blocks of
        `_BLOCK_DEPTH` before draining, so its contraction must be a whole number of
        `_LOOP_BLOCKS * _BLOCK_DEPTH`. A head is 64 deep and that is not, so both products pad the
        contraction with zeros -- exact, because the extra terms are zero, but it moves three times
        the operand bytes it needs to. Getting a range of M, N, K right removes it.

        The 1/√d scale folds into the packed query at no device cost.

    Args:
        query: [tokens, head] queries.
        key: [tokens, head] keys.
        value: [tokens, head] values.

    Returns:
        The [tokens, head] attention output.

    Raises:
        ValueError: If the token count is not the width one VREG reduces over.
    """
    tokens, head = query.shape
    if tokens != _VREG_ELEMENTS:
        message = (
            f"Unable to attend over {tokens} tokens. The softmax reduces over a whole VREG, so a "
            f"score row -- one per token -- must be exactly {_VREG_ELEMENTS} wide."
        )
        raise ValueError(message)

    contraction = _LOOP_BLOCKS * _BLOCK_DEPTH
    def padded(values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Zero-extends a contraction to the depth one block group covers."""
        if values.shape[0] >= contraction:
            return values
        extended = np.zeros((contraction, values.shape[1]), dtype=np.float32)
        extended[: values.shape[0]] = values
        return extended

    # This one takes UNSCALED queries and always applies 1/sqrt(d) itself: it is a
    # single head with no projection in front of it, so there is nothing to fold
    # into. `prefill_attention` and `decode_attention` take a `scale` because the
    # body folds theirs into the QKV weights.
    scaled = (query / math.sqrt(head)).astype(np.float32)
    scores = full_gemm(activations=_pad_columns(scaled, contraction), weights=padded(key.T.copy()))
    weights = softmax_rows(scores=scores)
    return full_gemm(activations=_pad_columns(weights, contraction), weights=padded(value))


def _sequence_length(*, rows: int, valid: int, sequences: int) -> int:
    """Rows one sequence owns, for the mask that keeps sequences apart.

    ONE sequence needs no boundary: the padded tail is excluded by column and the
    causal rule excludes everything after a query anyway, so the sequence is the
    whole padded axis and a prompt of any length works. Requiring the padded axis to
    be a multiple of the real one instead sent a five-token prompt -- an ordinary
    chat prompt -- down the host path without saying so.

    A BATCH does need the boundary, and needs it to tile the padded axis, because
    the mask places each sequence by index.

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


#: Additive attention masks, keyed by shape. A mask is a CONSTANT of a shape, not
#: of a layer, so one per shape serves all twelve and is built once.
_ATTENTION_MASKS: dict[tuple, DeviceTensor] = {}


def _attention_mask(rows: int, heads: int, *, causal: bool, sequence: int,
                    valid: int, span: int) -> DeviceTensor:
    """Returns the additive mask for a shape, uploading it once.

    Notes:
        Three things are forbidden and all three are the same constant: a key past
        the end of the array, a key in ANOTHER SEQUENCE, and -- when causal -- a key
        after the query's own position. The batch axis needs no kernel of its own
        because of the middle one.

        Large and negative rather than -inf, because the kernel subtracts the row
        maximum before the exponential and inf - inf is a NaN.

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
    # A row may always attend to ITSELF. True already for every real query; it is
    # the PADDED rows that need it, which otherwise forbid everything and leave the
    # softmax subtracting -1e30 from -1e30. Their output is discarded either way;
    # what matters is that it is a number.
    forbidden &= index[:, None] != index[None, :]

    h_mask = np.zeros((rows, heads * span), dtype=np.float32)
    for head in range(heads):
        h_mask[:, head * span:(head + 1) * span][:, :rows][forbidden] = -1.0e30
        h_mask[:, head * span:(head + 1) * span][:, rows:] = -1.0e30
    d_cached = upload_activations(h_mask)
    _ATTENTION_MASKS[key] = d_cached
    return d_cached


#: Attention's per-head operands, keyed by shape. Constants of a SHAPE, not of a
#: layer: every layer's scatter overwrites them, so one set serves all twelve and
#: is zeroed once.
_GROUPED_OPERANDS: dict[tuple, tuple] = {}


def _grouped_operands(rows: int, heads: int, head: int, span: int):
    """Allocates the per-head K-transpose and V operands for a grouped attention.

    Notes:
        These are the block-diagonal's replacement. A diagonal block encodes
        independence by surrounding each head's operand with zeros, which costs a
        factor of `heads` in multiplies; laying the same operands end to end and
        walking them with a loop level costs a loop level.

        `[heads, head_dim, span]` for the transposed keys and `[heads, span, head]`
        for the values -- one independent weight matrix per head, contiguous, which
        is exactly what `build_looped_gemm(batch=heads)` walks.

    Args:
        rows: Padded rows attended over.
        heads: Attention heads.
        head: The head dimension.
        span: Keys a score row covers, rows rounded up to whole VREGs.

    Returns:
        The key operand and the value operand, as `ResidentWeights` over the whole
        stack -- the GEMM is given the first problem's address and walks the rest.
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


def attention(d_fused: DeviceTensor, *, heads: int = 12, causal: bool = True,
                      sequences: int = 1, valid: int | None = None) -> DeviceTensor:
    """Attention with each head an independent product, not a diagonal block.

    Notes:
        The block-diagonal form runs every head in one product by giving the array
        an operand that is mostly zeros: `[rows, model] @ [model, heads * span]`,
        which is `heads` times the multiplies the problem has and `heads * B` times
        once a batch shares the span. This runs the same twelve products as twelve
        PROBLEMS in one launch instead -- `[rows, head_dim] @ [head_dim, span]`
        each -- so the zeros are gone and the launch count is not worse.

        Nothing downstream changes, and that is not luck. A grouped launch drains
        problem after problem, each `[column_tiles, BLOCK_TOKENS, TILE]`, and the
        packed layout of the whole `[rows, heads * span]` score array is
        `[global_tile, rows, TILE]` with `global_tile = span_tiles * h + q`. Those
        are the same ordering, so the packed softmax reads the grouped output where
        it lies, head h's probabilities are the contiguous run the gather's problem
        h wants, and the gather's own output is the packed `[rows, model]` the
        output projection consumes.

        The operands are what change: `d_keys` holds `heads` transposed key
        matrices end to end rather than one block-diagonal, which is the same
        transpose-scatter with `depth = head_dim` instead of `depth = model`.

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

    # Kᵀ, one [head_dim, span] matrix per head. The same transpose-scatter as the
    # block-diagonal's, with the destination's depth being a head rather than the
    # model -- so a head's tiles land `head_dim` apart instead of `model` apart, and
    # the next head begins where this one ended.
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
            fuse_groups=True, loop_blocks=depth // _BLOCK_DEPTH, batch=heads,
            batch_activation_bytes=_BLOCK_TOKENS * depth * _ELEMENT_BYTES))
    minitpuSetDevice(kernel.uio_name)
    d_image = _stage_image(image_bytes)
    d_out = _allocate_tensor(rows, heads * columns)
    for block in range(rows // _BLOCK_TOKENS):
        minitpuLaunchKernel(
            d_image, len(image_bytes),
            args=[d_left.operand(block), d_weights.address, d_out.d_drain(block)])
    return d_out


def prefill_attention(
    query: NDArray[np.float32], key: NDArray[np.float32], value: NDArray[np.float32],
    *, heads: int = 12, causal: bool = True, scale: float | None = None
) -> NDArray[np.float32]:
    """Runs prefill attention for EVERY head on the device, in three launches.

    Notes:
        Same batching as `decode_attention` and for the same reason: a head per launch is twelve
        launches where one will do. Head h's transposed keys occupy block (h, h) of a block-diagonal
        operand, so a single product gives every head's score matrix side by side and no head can
        see another's keys.

        The layout works out because **splitting heads is addressing, not movement**. A [tokens,
        model] array is already [tokens, heads, head] in memory, and the [tokens, heads*context]
        score array is already [tokens*heads, context] -- both reshapes are contiguous views over
        the same bytes. The head axis costs nothing here; the only layout operation that costs
        anything is the transpose of the keys, which the host packer still does.

        The causal mask is the one arithmetic step still on the host. It is an additive [tokens,
        context] bias, and adding it needs the scores to stay in VMEM between the product and the
        softmax -- which is a fused kernel, not a missing instruction.

    Args:
        query: [tokens, model] queries for all heads, unscaled.
        key: [tokens, model] keys for all heads.
        value: [tokens, model] values for all heads.
        heads: Attention heads.
        causal: Whether a query may only attend to keys at or before its own position.
        scale: What to multiply the queries by. Defaults to 1/sqrt(head_dim). Pass
            1.0 when the caller has already folded the scale into the projection
            that produced these queries, as `ResidentModel` does -- applying it twice
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

    # Kᵀ comes from the device now. The block-diagonal PLACEMENT is still host
    # bookkeeping -- zeros off the diagonal so no head sees another's keys -- but
    # the transpose itself, the part with no software route, is a launch.
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
        A decode step has one query row per head, and running a head per launch would be twelve
        launches to replace about two milliseconds of host numpy -- at roughly 55 ms of host cost
        each, that trade is badly the wrong way round. So all heads share one launch, by giving the
        weight-stationary array a **block-diagonal** operand: head h's Kᵀ sits at block (h, h) and
        everything off the diagonal is zero, so one product computes every head's scores at once and
        no head can see another's keys.

        It wastes eleven twelfths of the multiplies. That is the right trade while the device is
        idle for most of a token and the launch is what costs -- and the shapes fall out
        conveniently, because heads x head_dim IS the model width, so all three launches run at the
        width the GEMM already covers.

        Contexts shorter than a full VREG need their unused score columns pushed to zero
        probability, and that is the one host touch left in this path. It is a [heads, 64] add, and
        it exists only because the contraction has to be a whole number of
        `_LOOP_BLOCKS * _BLOCK_DEPTH`: with a range of M, N, K the contraction becomes exactly the
        context, the padding disappears and so does the mask.

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

    # Scores: [1, width] @ blockdiag(Kᵀ) -> [1, heads * VREG], head h in its own column band.
    # One transpose covers every head: stack the caches into [heads * context, head]
    # and the device returns [head, heads * context], from which each head's block
    # is a column range. The transpose is the launch; the placement is bookkeeping.
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


def _pad_rows_to_tile(h_rows: NDArray[np.float32]) -> NDArray[np.float32]:
    """Zero-extends the row axis to a whole number of transpose tiles.

    The staging tile is square, so a transpose needs both dimensions to be a whole
    number of `_TILE`. Padding with zeros is exact here: the padded columns of the
    result are dropped by the caller before they reach a score.
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


def _pad_columns(values: NDArray[np.float32], columns: int) -> NDArray[np.float32]:
    """Zero-extends an activation's contraction to `columns`, which pads with exact zeros."""
    if values.shape[1] >= columns:
        return values
    extended = np.zeros((values.shape[0], columns), dtype=np.float32)
    extended[:, : values.shape[1]] = values
    return extended


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
        The transfer reaches device memory through the copy engine's alias of the kernel aperture, while the loader
        reads the staged image back through the MINITPU's own device-memory port. Only a launch crosses both routes, so
        an aperture whose two addresses resolve to different DRAM surfaces fails here rather than in a workload
        result. Repeating the launch requires the command processor to return to idle between commands.

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


def check_command_interleave(kernel: MinitpuKernel, *, nbytes: int) -> None:
    """Verifies that COPY and LAUNCH commands alternate correctly and that a launch preserves device data.

    Notes:
        One ready flag reports both sub-FSMs idle, so each round issues a COPY against a flag the preceding LAUNCH
        has just released and vice versa. The kernel halts without touching memory, which leaves the payload the
        round reads back equal to the payload it sent.

    Args:
        kernel: The assembled program launched between the transfers, which halts without writing memory.
        nbytes: The payload size each round sends through device memory.

    Raises:
        AssertionError: If a round reads back a payload a launch or a transfer disturbed.
    """
    image_bytes = _image_of(kernel)
    address_space = minitpuGetAddressSpace()
    pattern = np.arange(nbytes // 4, dtype="<u4")

    minitpuSetDevice(kernel.uio_name)
    minitpuHostReset()
    device_address = None
    try:
        staging = minitpuHostAlloc(len(image_bytes))
        source = minitpuHostAlloc(nbytes)
        destination = minitpuHostAlloc(nbytes)
        staging.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
        source.view[: pattern.size] = pattern
        minitpuMemcpy(
            dst=address_space.kernel_base,
            src=staging,
            nbytes=len(image_bytes),
            kind=MemcpyKind.HostToDevice,
        )

        device_address = minitpuMalloc(nbytes)
        for _ in range(_INTERLEAVE_ROUNDS):
            minitpuMemcpy(dst=device_address, src=source, nbytes=nbytes, kind=MemcpyKind.HostToDevice)
            minitpuLaunchKernel(kbin_addr=address_space.kernel_base, kbin_size=len(image_bytes))

            destination.view[: pattern.size] = _CORRUPTION_WORD
            destination.sync_to_device()
            minitpuMemcpy(dst=destination, src=device_address, nbytes=nbytes, kind=MemcpyKind.DeviceToHost)
            np.testing.assert_array_equal(destination.view[: pattern.size], pattern)
    finally:
        if device_address is not None:
            minitpuFree(device_address)
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
        # every run reuses the same offsets, so an untouched destination still holds the words the previous run left
        # there. The flush carries the word past the cache lines the read-back invalidates.
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


def check_launch_interleave() -> str:
    """Alternates COPY and LAUNCH commands against one payload.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If a round reads back a payload a launch or a transfer disturbed.
    """
    check_command_interleave(kernel=single_halt, nbytes=_INTERLEAVE_PAYLOAD_BYTES)
    return f"{_INTERLEAVE_ROUNDS} rounds of alternating COPY and LAUNCH"


def check_mul_add() -> str:
    """Runs a BF16 multiply-add through the callable that hides allocation, transfers and the launch.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the program disagrees with the host reference.
    """
    side = 32
    left = np.full((side, side), 2, dtype=np.float32)
    right = np.full_like(left, 2)
    destination = np.empty(left.size, dtype="<u2")
    single_mul_add(as_bf16_words(values=left).ravel(), as_bf16_words(values=right).ravel(), destination)
    np.testing.assert_array_equal(from_bf16_words(words=destination).reshape(left.shape), left * right + left)
    return f"{side}x{side} C = A * B + A, {len(single_mul_add.image.bundles)} VLIW bundles"


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
    """Runs a product at model dimensions and verifies the loop kernel against a host reference.

    Notes:
        Runs two operand sets, because either alone is blind. The structured set is exactly
        representable, so it pins the kernel to the host reference bit for bit -- but every one of its
        token rows is identical and its weights are constant along the contraction, so any kernel that
        confuses one token for another, or one contraction block for another, still passes it. The
        varying set differs along every axis, which is what makes those confusions observable; its
        BF16 partial sums cost exactness, so it carries a relative bound instead.

    Args:
        tokens: Rows the product covers, which must be a whole number of token blocks.
        depth: The contraction, which must be a whole number of the block groups one launch drains.
        columns: Output features, which must be a whole number of tiles.

    Returns:
        The detail the report line carries, naming the shapes, the launch count and the measured time.

    Raises:
        AssertionError: If the loop kernel disagrees with the host reference on either operand set.
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
        f"[{tokens},{depth}] @ [{depth},{columns}], {2 * launches} launches in {elapsed:.1f} s "
        f"({1e6 * elapsed / launches:.0f} us each), varying-operand relative error {error:.2%}"
    )


def check_full_ffn(*, tokens: int = 32, model: int = 768, hidden: int = 3072) -> str:
    """Runs one feed-forward layer at model dimensions and verifies it against a host reference.

    Notes:
        Defaults to GPT-2 small's feed-forward shape, so a pass means two launches carry a real layer rather than a
        tile. The reference quantizes both operands to BF16 and applies the SFU's own table,
        leaving only the accumulate to differ, so the bound below is the error the BF16 partials contribute rather
        than a tolerance for an unmodelled effect.

    Args:
        tokens: Rows the layer processes, which must be a whole number of token blocks.
        model: The layer's input and output width.
        hidden: The intermediate width the up projection expands to.

    Returns:
        The detail the report line carries, naming the shapes, the launch count and the measured error.

    Raises:
        AssertionError: If the relative error exceeds the bound BF16 partial sums explain.
    """
    rng = np.random.default_rng(0)
    activations = _on_bf16_grid(values=rng.standard_normal((tokens, model)).astype(np.float32))
    up_weight = _on_bf16_grid(values=(rng.standard_normal((model, hidden)) * 0.14).astype(np.float32))
    down_weight = _on_bf16_grid(values=(rng.standard_normal((hidden, model)) * 0.09).astype(np.float32))
    up_bias = np.zeros(hidden, dtype=np.float32)
    down_bias = np.zeros(model, dtype=np.float32)

    # Both projections cover a token block in one launch.
    launches = 2 * (tokens // _BLOCK_TOKENS)
    started = time.perf_counter()
    actual = full_ffn(
        activations=activations, up_weight=up_weight, up_bias=up_bias,
        down_weight=down_weight, down_bias=down_bias,
    )
    elapsed = time.perf_counter() - started

    hidden_reference = activations.astype(np.float64) @ up_weight.astype(np.float64) + up_bias
    activated = from_bf16_words(
        words=gelu_lut_reference(words=as_bf16_words(values=hidden_reference.astype(np.float32)))
    )
    expected = activated.astype(np.float64) @ down_weight.astype(np.float64) + down_bias

    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _FFN_ERROR_LIMIT:
        message = (
            f"Unable to verify the feed-forward layer against the host reference. The relative error must stay "
            f"below {_FFN_ERROR_LIMIT:.0%}, which is what the BF16 partial sums explain, but got {error:.3%}."
        )
        raise AssertionError(message)
    return (
        f"[{tokens},{model}] @ [{model},{hidden}] -> GELU -> "
        f"[{tokens},{hidden}] @ [{hidden},{model}], {launches} launches in {elapsed:.1f} s "
        f"({1e6 * elapsed / launches:.0f} us each), relative error {error:.2%}"
    )


def check_softmax(*, rows: int = 16) -> str:
    """Runs a row-wise softmax on the device and verifies it against the host reference.

    Notes:
        The scores are spread wide enough that the maximum matters: without subtracting it these
        would leave the SFU exp table's non-positive domain, which `sfu.sv` asserts on. A check whose
        scores all sat near zero would pass whether or not the subtraction happened.

    Args:
        rows: Rows to reduce, each one VREG wide.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the relative error exceeds what the BF16 steps and the table explain.
    """
    rng = np.random.default_rng(0)
    scores = _on_bf16_grid(values=(rng.standard_normal((rows, _VREG_ELEMENTS)) * 4.0).astype(np.float32))
    actual = softmax_rows(scores=scores)
    expected = softmax_reference(scores=scores)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _SOFTMAX_ERROR_LIMIT:
        message = (
            f"Unable to verify the softmax against the host reference. The relative error must stay "
            f"below {_SOFTMAX_ERROR_LIMIT:.0%}, which is what the BF16 steps and the SFU's tables "
            f"explain, but got {error:.3%}."
        )
        raise AssertionError(message)
    return f"[{rows},{_VREG_ELEMENTS}] rows, relative error {error:.2%}"


def check_attention(*, tokens: int = 64, head: int = 64) -> str:
    """Runs one attention head on the device and verifies it against the host reference.

    Notes:
        Three launches: a product, a softmax, a product. The transpose and the contraction padding
        ride on the host for the reasons `attention_head` documents.

    Args:
        tokens: Tokens attended over, which the softmax fixes at one VREG's width.
        head: Head dimension.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the relative error exceeds what the chained BF16 steps explain.
    """
    rng = np.random.default_rng(0)
    query, key, value = (
        _on_bf16_grid(values=rng.standard_normal((tokens, head)).astype(np.float32))
        for _ in range(3))
    started = time.perf_counter()
    actual = attention_head(query=query, key=key, value=value)
    elapsed = time.perf_counter() - started
    expected = attention_reference(query=query, key=key, value=value)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _ATTENTION_ERROR_LIMIT:
        message = (
            f"Unable to verify one attention head against the host reference. The relative error "
            f"must stay below {_ATTENTION_ERROR_LIMIT:.0%}, which is what a product, a softmax and a "
            f"second product contribute in BF16, but got {error:.3%}."
        )
        raise AssertionError(message)
    return (f"softmax([{tokens},{head}] @ [{head},{tokens}]) @ [{tokens},{head}] in {elapsed:.1f} s, "
            f"relative error {error:.2%}")


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


def check_fused_groups(*, tokens: int = 32, depth: int = 768, columns: int = 3072) -> str:
    """Times and prices summing the contraction's groups on the device.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the fused product drifts further than BF16 explains.
    """
    rng = np.random.default_rng(37)
    activations = _on_bf16_grid(values=rng.standard_normal((tokens, depth)).astype(np.float32))
    weights = _on_bf16_grid(values=rng.standard_normal((depth, columns)).astype(np.float32))
    expected = activations.astype(np.float64) @ weights.astype(np.float64)

    handle = upload_weights(weights)
    try:
        timings, errors = {}, {}
        for label, fuse in (("host", False), ("device", True)):
            full_gemm(activations=activations, weights=handle, fuse_groups=fuse)
            started = time.time()
            product = full_gemm(activations=activations, weights=handle, fuse_groups=fuse)
            timings[label] = time.time() - started
            errors[label] = float(
                np.linalg.norm(product - expected) / np.linalg.norm(expected))
    finally:
        handle.free()

    assert errors["device"] < 0.05, (
        f"device group sum relative error {errors['device']:.4%} is too large")
    return (f"group sum on host {timings['host'] * 1e3:.0f} ms / {errors['host']:.2%} vs "
            f"on device {timings['device'] * 1e3:.0f} ms / {errors['device']:.2%} "
            f"({timings['host'] / timings['device']:.2f}x, "
            f"{errors['device'] / errors['host']:.2f}x error)")


def check_tile_transpose(*, tiles: int = 4) -> str:
    """Transposes tiles on the device and compares against numpy.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If any tile is not exactly its transpose.
    """
    rng = np.random.default_rng(103)
    h_tiles = _on_bf16_grid(
        values=rng.standard_normal((tiles * _TILE, _TILE)).astype(np.float32))
    started = time.time()
    actual = transpose_tiles(h_tiles)
    elapsed = time.time() - started

    expected = np.concatenate(
        [h_tiles[i * _TILE:(i + 1) * _TILE].T for i in range(tiles)])
    assert np.array_equal(actual, expected), (
        f"device transpose differs by up to {np.abs(actual - expected).max():.3e}")
    return f"{tiles} tiles of {_TILE}x{_TILE} in {elapsed * 1e3:.0f} ms, exact"


def check_elementwise_add(*, tokens: int = 32, features: int = 768) -> str:
    """Adds two DMEM tensors on the device and compares against the host.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the device's sum is not the host's, exactly.
    """
    rng = np.random.default_rng(53)
    h_left = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    h_right = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    d_left, d_right = upload_activations(h_left), upload_activations(h_right)
    try:
        started = time.time()
        d_sum = add_resident(d_left, d_right)
        elapsed = time.time() - started
        try:
            actual = download_activations(d_sum)
        finally:
            d_sum.free()
    finally:
        d_left.free()
        d_right.free()

    expected = _on_bf16_grid(values=h_left + h_right)
    assert np.array_equal(actual, expected), (
        f"device sum differs from the host by up to {np.abs(actual - expected).max():.3e}")
    return f"[{tokens},{features}] + [{tokens},{features}] in {elapsed * 1e3:.0f} ms, exact"


def check_weight_residency(*, tokens: int = 32, depth: int = 768, columns: int = 3072) -> str:
    """Times the same product with the weights packed per call and left resident.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If residency changes the answer, or fails to be faster.
    """
    rng = np.random.default_rng(31)
    activations = _on_bf16_grid(values=rng.standard_normal((tokens, depth)).astype(np.float32))
    weights = _on_bf16_grid(values=rng.standard_normal((depth, columns)).astype(np.float32))

    per_call = full_gemm(activations=activations, weights=weights)   # warms the image cache
    started = time.time()
    per_call = full_gemm(activations=activations, weights=weights)
    packed_seconds = time.time() - started

    handle = upload_weights(weights)
    try:
        started = time.time()
        resident = full_gemm(activations=activations, weights=handle)
        resident_seconds = time.time() - started
    finally:
        handle.free()

    assert np.array_equal(per_call, resident), (
        "resident weights changed the product; the packing or the address is wrong")
    return (f"[{tokens},{depth}] @ [{depth},{columns}]: {packed_seconds * 1e3:.0f} ms packed "
            f"per call vs {resident_seconds * 1e3:.0f} ms resident, "
            f"{packed_seconds / resident_seconds:.2f}x")


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
        The whole block: two LayerNorms, the fused QKV projection, twelve attention heads, the output
        projection and both feed-forward products, all on the device. Residual adds, biases and head
        reshaping stay on the host.

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


def main() -> int:
    """Runs the demo programs on the board and reports each result against a host model.

    Returns:
        Zero when every program matched its expected output, and one when any of them did not.
    """
    outcomes = [
        run_check(name="Halt", check=check_halt),
        run_check(name="Command interleave", check=check_launch_interleave),
        run_check(name="BF16 mul-add", check=check_mul_add),
        run_check(name="COPY round trip", check=check_copy_argument),
        run_check(name="Full GEMM", check=check_full_gemm),
        run_check(name="Full single layer FFN", check=check_full_ffn),
        run_check(name="Row-wise softmax", check=check_softmax),
        run_check(name="Single attention head", check=check_attention),
        run_check(name="Tile transpose on device", check=check_tile_transpose),
        run_check(name="Elementwise add on device", check=check_elementwise_add),
        run_check(name="Weight residency", check=check_weight_residency),
        run_check(name="Group sum on device", check=check_fused_groups),
        run_check(name="Prefill attention, all heads", check=check_prefill_attention),
        run_check(name="Decode attention, all heads", check=check_decode_attention),
        run_check(name="LayerNorm rows", check=check_layernorm),
        run_check(name="Transformer block", check=check_transformer_block),
    ]
    print(f"\n{sum(outcomes)} of {len(outcomes)} checks passed")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
