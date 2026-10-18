"""Provides board bring-up and the process-wide host and device memory allocators.

Each pool covers one aperture, the host one by bumping and the device one by first fit, and neither frees a
single region.
"""

from __future__ import annotations

from .address_space import DeviceAddressSpace, minitpuConfigureAddressSpace, minitpuGetAddressSpace
from .backend import HostRegion, UdmabufBuffer, UioMmio, reset_uio_cache
from .memcpy import CDMA_ALIGN, _set_backend, _set_uio_name

_DEFAULT_UDMABUF_INDEX: int = 0
"""The /dev/udmabufN opened when the caller expresses no preference."""

_UDMABUF: UdmabufBuffer | None = None
"""The one host DMA buffer cached process-wide, opened lazily on the first host allocation."""

_UDMABUF_INDEX: int | None = None
"""Which /dev/udmabufN the cached buffer is, so a call naming another index is rejected instead of served."""

_UDMABUF_OWNED: bool = False
"""Determines whether this allocator opened the cached buffer and may therefore close it."""

_BUMP_OFFSET: int = 0
"""Byte offset the next host region is carved at."""

_DEVICE_POOL: tuple[int, int] | None = None
"""Base and size of the DMEM span the device allocator serves."""

_DEVICE_FREE_SPANS: list[tuple[int, int]] = []
"""Free base and size spans of the DMEM pool, kept sorted and coalesced."""

_DEVICE_ALLOCATIONS: dict[int, int] = {}
"""Byte size of every live device allocation, keyed by the pointer handed out."""


def open_device(
    uio_name: str,
    *,
    udmabuf_index: int = 0,
    address_space: DeviceAddressSpace | None = None,
):
    """Opens the MINITPU UIO register window and a udmabuf host buffer, installing both process-wide.

    Args:
        uio_name: The UIO node name that exposes the board's AXI-Lite CMD window.
        udmabuf_index: The udmabuf instance to open as the host DMA buffer.
        address_space: The board map to install, defaulting to the map already configured.

    Returns:
        The register backend and the host DMA buffer, the latter for buffer-level use such as capacity checks and
        whole-buffer cache sync. Carving a region off it by hand overlaps the ones ``minitpuHostAlloc`` hands out.
    """
    if address_space is not None:
        minitpuConfigureAddressSpace(address_space)
    # Rediscovers the node, since the PL may have been reprogrammed since this process last looked, which moves
    # the UIO index.
    reset_uio_cache()
    mmio = UioMmio.open_by_name(uio_name)
    _set_uio_name(uio_name)
    _set_backend(mmio)
    buffer = UdmabufBuffer.open(udmabuf_index)
    _set_udmabuf(buffer, index=udmabuf_index)
    return mmio, buffer


def minitpuSetDevice(uio_name: str) -> None:
    """Selects the MINITPU device to drive, opening and installing its register window alone.

    Host allocation is left untouched, matching CUDA pinned host memory, which survives a device switch.

    Args:
        uio_name: The UIO node name that exposes the board's AXI-Lite CMD window.

    Raises:
        RuntimeError: When no UIO node carries that name, which is the off-board case.
    """
    mmio = UioMmio.open_by_name(uio_name)
    _set_uio_name(uio_name)
    _set_backend(mmio)


def minitpuHostAlloc(nbytes: int, *, udmabuf_index: int | None = None) -> HostRegion:
    """Carves the next pinned host DMA region out of the one cached udmabuf.

    This is the single source of host regions, since a region carved by hand off an ``open_device`` buffer starts
    from offset zero and overlaps the ones handed out here. Every region is kept ``CDMA_ALIGN`` aligned, matching
    the memcpy alignment contract on the host address.

    Args:
        nbytes: The byte size of the region, rounded up to the alignment before the cursor advances.
        udmabuf_index: The udmabuf to open, honoured only on the call that actually opens it.

    Raises:
        RuntimeError: When no udmabuf is available off-board, the index names a buffer other than the cached one,
            or the aligned request would run past the end of the buffer.
    """
    global _UDMABUF, _BUMP_OFFSET, _UDMABUF_OWNED

    if _UDMABUF is None:
        index = _DEFAULT_UDMABUF_INDEX if udmabuf_index is None else udmabuf_index
        _set_udmabuf(UdmabufBuffer.open(index), index=index)
        _UDMABUF_OWNED = True
    elif udmabuf_index is not None and udmabuf_index != _UDMABUF_INDEX:
        message = (
            f"Unable to allocate host memory from udmabuf{udmabuf_index}. This runtime caches exactly one udmabuf "
            f"process-wide and udmabuf{_UDMABUF_INDEX} is already cached, so call minitpuHostReset() before "
            f"allocating from a different udmabuf."
        )
        raise RuntimeError(message)

    aligned_size = _round_up(nbytes, CDMA_ALIGN)
    if _BUMP_OFFSET + aligned_size > _UDMABUF.size:
        # Names the cached buffer rather than the requested index, since the cached one is the truth.
        message = (
            f"Unable to allocate host memory. The udmabuf{_UDMABUF_INDEX} buffer of {_UDMABUF.size} bytes is "
            f"exhausted at offset {_BUMP_OFFSET}, but got a request for {nbytes} bytes, {aligned_size} aligned."
        )
        raise RuntimeError(message)

    words = _round_up(nbytes, 4) // 4  # Sizes the region in the uint32 words its view holds.
    region = HostRegion(_UDMABUF, _BUMP_OFFSET, words)
    _BUMP_OFFSET += aligned_size
    return region


def minitpuHostReset() -> None:
    """Drops the cached udmabuf and resets the host allocator's bump offset.

    The mapping is closed only when the allocator opened it itself, since a buffer installed by ``open_device``
    belongs to its installer. Any region carved from a buffer this call does close is dead afterwards.
    """
    _set_udmabuf(None)


def minitpuHostRewind() -> None:
    """Resets the host allocator's bump offset while keeping the udmabuf mapped.

    A reset re-mmaps sixteen megabytes on the next allocation, profiled at 3.7 ms a layer. It also differs in the
    harder direction: a region held across a reset points at an unmapped page and faults, while one held across a
    rewind is silently aliased by a later allocation. Use this only where regions are consumed immediately.
    """
    global _BUMP_OFFSET
    _BUMP_OFFSET = 0


def minitpuMalloc(nbytes: int) -> int:
    """Allocates device memory first-fit and returns the aligned physical DMEM pointer.

    Args:
        nbytes: The byte size of the allocation, rounded up to ``CDMA_ALIGN``.

    Raises:
        ValueError: When the requested size is not positive.
        MemoryError: When no free span can satisfy the request.
    """
    if nbytes <= 0:
        message = f"Unable to allocate device memory. The requested size must be positive, but got {nbytes}."
        raise ValueError(message)
    _ensure_device_pool()
    aligned_size = _round_up(int(nbytes), CDMA_ALIGN)
    for index, (base, size) in enumerate(_DEVICE_FREE_SPANS):
        if size >= aligned_size:
            if size == aligned_size:
                _DEVICE_FREE_SPANS.pop(index)
            else:
                _DEVICE_FREE_SPANS[index] = (base + aligned_size, size - aligned_size)
            _DEVICE_ALLOCATIONS[base] = aligned_size
            return base
    message = (
        f"Unable to allocate device memory. The DMEM aperture needs {aligned_size} free contiguous bytes, but the "
        f"largest free span is {max((size for _, size in _DEVICE_FREE_SPANS), default=0)} bytes."
    )
    raise MemoryError(message)


def minitpuFree(address: int) -> None:
    """Returns a device pointer's span to the free-list, coalescing it with any adjacent free span.

    Raises:
        KeyError: When the pointer was not handed out by this allocator.
    """
    if address not in _DEVICE_ALLOCATIONS:
        message = f"Unable to free device memory. The pointer 0x{address:x} was not returned by minitpuMalloc."
        raise KeyError(message)
    size = _DEVICE_ALLOCATIONS.pop(address)
    _DEVICE_FREE_SPANS.append((address, size))
    _DEVICE_FREE_SPANS.sort()
    merged: list[tuple[int, int]] = []
    for base, span in _DEVICE_FREE_SPANS:
        if merged and merged[-1][0] + merged[-1][1] == base:
            merged[-1] = (merged[-1][0], merged[-1][1] + span)
        else:
            merged.append((base, span))
    _DEVICE_FREE_SPANS[:] = merged


def minitpuDeviceReset() -> None:
    """Resets the device allocator to a single free aperture span, without touching device memory."""
    global _DEVICE_POOL
    _DEVICE_POOL = _device_pool()
    _DEVICE_FREE_SPANS[:] = [_DEVICE_POOL]
    _DEVICE_ALLOCATIONS.clear()


def minitpuInAperture(address: int, nbytes: int = 0) -> bool:
    """Returns True when the given range lies inside the DMEM aperture.

    The device has no hardware bounds check, so a pointer bound as a kernel argument is validated with this before
    a launch. A bad pointer walks off the buffer silently on hardware.
    """
    base, size = _device_pool()
    return base <= address and (address + max(0, nbytes)) <= (base + size)


def _set_udmabuf(buffer: UdmabufBuffer | None, *, index: int = _DEFAULT_UDMABUF_INDEX) -> None:
    """Installs the cached udmabuf the host allocator carves from, or clears the cache when given None.

    Only a buffer this allocator opened itself is closed on replacement, since an installed buffer belongs to its
    installer, who may still hold live regions into it.
    """
    global _UDMABUF, _BUMP_OFFSET, _UDMABUF_INDEX, _UDMABUF_OWNED
    if _UDMABUF is not None and _UDMABUF_OWNED and _UDMABUF is not buffer:
        _UDMABUF.close()
    _UDMABUF = buffer
    _UDMABUF_INDEX = index if buffer is not None else None
    _UDMABUF_OWNED = False
    _BUMP_OFFSET = 0


def _round_up(value: int, align: int) -> int:
    """Rounds a value up to the next multiple of an alignment."""
    return (value + align - 1) // align * align


def _device_pool() -> tuple[int, int]:
    """Returns the base and size of the configured board's kernel-visible data range.

    The aperture's low reserved prefix belongs to instruction-image staging, so the pool is the remainder.
    """
    space = minitpuGetAddressSpace()
    return space.kernel_base + space.reserved, space.kernel_size - space.reserved


def _ensure_device_pool() -> None:
    """Points the device allocator at the configured board's pool.

    Raises:
        RuntimeError: When the configured address space changed while allocations are live.
    """
    global _DEVICE_POOL
    pool = _device_pool()
    if _DEVICE_POOL == pool:
        return
    if _DEVICE_ALLOCATIONS:
        message = "Unable to change the device address space. The allocator still holds live allocations."
        raise RuntimeError(message)
    _DEVICE_POOL = pool
    _DEVICE_FREE_SPANS[:] = [pool]
