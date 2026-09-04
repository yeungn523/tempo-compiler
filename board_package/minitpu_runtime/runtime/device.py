"""Board bring-up: open a device's register window + host DMA buffer.

:func:`open_device` composes the runtime's existing HAL primitives into the one
sequence a board client needs before issuing COPY/LAUNCH commands:

  1. mmap the MINITPU AXI-Lite CMD window over a Linux UIO node
     (:meth:`UioMmio.open_by_name`),
  2. install it as the runtime's process-wide MMIO backend (via the private
     ``_set_backend`` hook -- there is no public backend setter) and bind the UIO
     node name for any later re-detection (``_set_uio_name``), and
  3. open a contiguous udmabuf host DMA buffer (:meth:`UdmabufBuffer.open`).

The UIO node name is a REQUIRED argument: the runtime is board-AGNOSTIC and
carries no default board node name, so the caller (a board test / a board
runner) supplies its board's node name.  Off-board this raises a clear
RuntimeError (no matching UIO / no udmabuf), not on import.

This module also carries the two CUDA-style user-facing entry points built on
top of that HAL:

  - :func:`minitpuSetDevice`  - open + install the register-window backend only
    (the ``cudaSetDevice`` analog).
  - :func:`minitpuHostAlloc`  - carve pinned host DMA regions out of one
    lazily-opened udmabuf (the ``cudaHostAlloc`` / ``cudaMallocHost`` analog).
  - :func:`minitpuHostReset`  - drop the cached udmabuf (test isolation / re-run).

``open_device`` remains the lower-level, all-in-one primitive used by board
tests and diagnostics; ``minitpuSetDevice`` intentionally does less (no
udmabuf) so device selection and host-buffer allocation are independent, CUDA-
style steps.

SINGLE OWNERSHIP: there is exactly ONE cached udmabuf per process.
``open_device`` installs the buffer it opens, so a following
:func:`minitpuHostAlloc` carves out of that SAME buffer rather than opening a
second, independently-mmap'd view of the same physical CMA block.

SINGLE SOURCE OF REGIONS: :func:`minitpuHostAlloc` is the only supported way to
get a ``HostRegion``.  ``open_device`` still returns its ``buf`` for
buffer-level use (capacity checks, phys reporting, whole-buffer cache sync), but
hand-carving ``HostRegion(buf, off, words)`` off it bypasses the allocator's
bump cursor and overlaps allocator regions.  See :func:`open_device`.
"""

from .address_space import (
    DeviceAddressSpace,
    minitpuConfigureAddressSpace,
    minitpuGetAddressSpace,
)
from .backend import HostRegion, UdmabufBuffer, UioMmio, reset_uio_cache
from .memcpy import CDMA_ALIGN, _set_backend, _set_uio_name


def open_device(
    uio_name: str,
    *,
    udmabuf_index: int = 0,
    address_space: DeviceAddressSpace | None = None,
):
    """Open the MINITPU UIO register window and a udmabuf host buffer."""
    if address_space is not None:
        minitpuConfigureAddressSpace(address_space)
    # An explicit open is the one place that must rediscover: the PL may have been
    # reprogrammed since this process last looked, which can move the UIO index.
    reset_uio_cache()
    mmio = UioMmio.open_by_name(uio_name)
    _set_uio_name(uio_name)
    _set_backend(mmio)
    buf = UdmabufBuffer.open(udmabuf_index)
    _set_udmabuf(buf, index=udmabuf_index)
    return mmio, buf


def minitpuSetDevice(uio_name: str) -> None:
    """Select the MINITPU device to drive (CUDA ``cudaSetDevice(int)`` analog).

    CUDA's ``cudaSetDevice(int device)`` selects among multiple GPUs by integer
    index, looked up against an internal device registry populated at driver
    init.  minitpu v1 targets a single MINITPU, so there is no registry and no
    index to select from: this function instead takes the board's UIO node
    name directly, and the runtime holds no hardcoded default -- the caller
    must supply it (mirroring :func:`open_device`).

    Multi-MINITPU support is [v2]: at that point this call switches to the CUDA
    shape (an integer ``device`` index), and the runtime grows an internal
    registry mapping index -> UIO node name, so user code selects a device by
    index exactly like CUDA. Until then, the node name IS the selection.

    Raw CSR/register access is intentionally NOT exposed through this (or any)
    user-facing runtime call: CUDA hides register-level device access behind
    the driver, and a management/telemetry surface (were one needed) would be
    a separate NVML-like API, not part of the compute host<->device path.
    Board diagnostics that need the raw MMIO handle use the lower-level
    :func:`open_device` (which returns ``(mmio, buf)``) instead.

    Opens the MINITPU AXI-Lite register window over ``uio_name``
    (:meth:`UioMmio.open_by_name`), binds the name (:func:`_set_uio_name`), and
    installs it as the process-wide MMIO backend (:func:`_set_backend`).
    Returns ``None``. Does NOT open a udmabuf -- that is
    :func:`minitpuHostAlloc`'s job, kept as an independent step so device
    selection and host-buffer allocation compose the way ``cudaSetDevice`` and
    ``cudaHostAlloc`` do.

    Does NOT reset the :func:`minitpuHostAlloc` udmabuf cache, and that is
    DELIBERATE, not an oversight: CUDA's pinned host memory is a process-wide
    resource that survives ``cudaSetDevice`` (a pinned allocation stays valid and
    mapped across device switches), so leaving the host buffer untouched is the
    faithful analog. Host allocation is reset only by the explicit
    :func:`minitpuHostReset`.

    Raises :class:`RuntimeError` off-board (no UIO node named ``uio_name``),
    inherited from :meth:`UioMmio.open_by_name`.
    """
    mmio = UioMmio.open_by_name(uio_name)
    _set_uio_name(uio_name)
    _set_backend(mmio)


# ---------------------------------------------------------------------------
# Pinned host DMA allocation (minitpuHostAlloc / minitpuHostReset)
# ---------------------------------------------------------------------------
# v1 host-allocation model: a bump allocator over ONE contiguous udmabuf,
# opened lazily on the first minitpuHostAlloc call and cached module-globally
# (mirrors the single process-wide MMIO backend in memcpy.py). Per-region free
# is [v2]; v1 only ever grows the offset, and minitpuHostReset drops the whole
# buffer for test isolation / re-runs.
_UDMABUF: UdmabufBuffer | None = None
_BUMP_OFFSET = 0
# Which /dev/udmabufN the cached buffer actually is (None when nothing cached).
# Tracked so a later call naming a DIFFERENT index is rejected loudly instead of
# silently served from the cached one, and so the exhaustion message can name the
# REAL buffer rather than whatever index the caller happened to pass.
_UDMABUF_INDEX: int | None = None
# True only when THIS allocator lazily opened the cached buffer, i.e. the runtime
# is its sole owner and may close() it on reset. False for a buffer installed via
# _set_udmabuf / open_device: the installer still holds it, so closing it would
# yank the mmap out from under live HostRegion views.
_UDMABUF_OWNED = False
# The /dev/udmabufN opened when the caller expresses no preference.
_DEFAULT_UDMABUF_INDEX = 0


def _set_udmabuf(buf: UdmabufBuffer | None, *, index: int = _DEFAULT_UDMABUF_INDEX) -> None:
    """Install the cached udmabuf used by :func:`minitpuHostAlloc` (private).

    Not exported. Used by :func:`open_device` to hand the runtime the buffer it
    opened, and by tests to inject a fake one. Pass ``None`` to clear the cache
    (equivalent to :func:`minitpuHostReset`) and force re-opening on the next
    call. ``index`` records which ``/dev/udmabufN`` ``buf`` corresponds to, so a
    later :func:`minitpuHostAlloc` naming a different index can be rejected.

    An installed buffer is NEVER closed by the runtime: the installer keeps
    ownership. Only a buffer this allocator lazily opened itself is closed on
    replacement/reset.
    """
    global _UDMABUF, _BUMP_OFFSET, _UDMABUF_INDEX, _UDMABUF_OWNED
    # Release the previous mmap only if we were its sole owner.
    if _UDMABUF is not None and _UDMABUF_OWNED and _UDMABUF is not buf:
        _UDMABUF.close()
    _UDMABUF = buf
    _UDMABUF_INDEX = index if buf is not None else None
    _UDMABUF_OWNED = False
    _BUMP_OFFSET = 0


def minitpuHostAlloc(nbytes: int, *, udmabuf_index: int | None = None) -> HostRegion:
    """Allocate pinned host DMA memory (CUDA ``cudaHostAlloc``/``cudaMallocHost`` analog).

    Returns a :class:`~runtime.backend.HostRegion` (a ``phys_addr`` plus
    a numpy ``uint32`` view and cache-sync hooks) usable directly as a
    :func:`~runtime.memcpy.minitpuMemcpy` host operand. This hides the
    udmabuf-open-plus-phys-carving dance a caller would otherwise repeat by hand.

    v1 allocation model: a bump allocator over ONE contiguous udmabuf. The
    first call opens (and caches, module-global) ``UdmabufBuffer.open(udmabuf_index)``;
    every call -- including the first -- then carves the next non-overlapping
    region out of it and advances a byte offset. Per-region free is [v2]; v1
    intentionally does not support it.

    ``udmabuf_index`` selects the buffer ONLY on the call that actually opens it
    (default ``None`` = no preference = ``udmabuf0``). Because v1 caches exactly
    one udmabuf process-wide, an EXPLICIT index naming a different buffer than
    the cached one cannot be honoured; rather than silently serving it from the
    cached buffer, that raises :class:`RuntimeError` pointing at
    :func:`minitpuHostReset`. Passing nothing always accepts whatever is cached,
    so ``open_device(uio, udmabuf_index=3)`` then ``minitpuHostAlloc(n)`` works:
    :func:`open_device` opened the buffer and its index is the one in force.

    This allocator is the SINGLE source of host regions: regions carved by hand
    off an :func:`open_device` buffer (``HostRegion(buf, 0, words)``) are
    invisible to it and WILL overlap the regions it hands out, since both start
    from offset 0. Allocate here instead of carving. See :func:`open_device`.

    Every region's phys offset is kept :data:`~runtime.memcpy.CDMA_ALIGN`
    (32-byte) aligned, matching :func:`minitpuMemcpy`'s alignment contract on
    ``host_phys`` (256-bit AXI CDMA, no DRE): ``nbytes`` is rounded up to that
    alignment before the bump offset advances, so back-to-back allocations
    never hand out an unaligned region.

    Raises :class:`RuntimeError` off-board (no udmabuf available), when
    ``udmabuf_index`` names a different buffer than the cached one, or when the
    udmabuf is exhausted (this request's aligned size would run past the end
    of the buffer).
    """
    global _UDMABUF, _BUMP_OFFSET, _UDMABUF_OWNED

    if _UDMABUF is None:
        index = _DEFAULT_UDMABUF_INDEX if udmabuf_index is None else udmabuf_index
        _set_udmabuf(UdmabufBuffer.open(index), index=index)
        _UDMABUF_OWNED = True  # we opened it, so we may close it on reset
    elif udmabuf_index is not None and udmabuf_index != _UDMABUF_INDEX:
        raise RuntimeError(
            f"minitpuHostAlloc: udmabuf{udmabuf_index} requested but "
            f"udmabuf{_UDMABUF_INDEX} is already cached; v1 caches exactly ONE "
            f"udmabuf process-wide. Call minitpuHostReset() before allocating "
            f"from a different udmabuf."
        )

    aligned_size = _round_up(nbytes, CDMA_ALIGN)
    if _BUMP_OFFSET + aligned_size > _UDMABUF.size:
        raise RuntimeError(
            # Name the CACHED buffer, not the requested index: past the first
            # call they are guaranteed equal, but the cached one is the truth.
            f"minitpuHostAlloc: udmabuf{_UDMABUF_INDEX} exhausted "
            f"(requested {nbytes} bytes, {aligned_size} aligned, at offset "
            f"{_BUMP_OFFSET}, buffer size {_UDMABUF.size})"
        )

    words = _round_up(nbytes, 4) // 4  # HostRegion.view is a uint32 array
    region = HostRegion(_UDMABUF, _BUMP_OFFSET, words)
    _BUMP_OFFSET += aligned_size
    return region


def minitpuHostReset() -> None:
    """Drop the cached udmabuf and reset the :func:`minitpuHostAlloc` bump offset.

    Needed for test isolation and re-running allocation from a clean state.
    The next :func:`minitpuHostAlloc` call re-opens ``udmabuf_index`` fresh.
    Per-region free is [v2]; this only resets the whole allocator, not a
    single region.

    Closes the cached buffer's mmap ONLY if :func:`minitpuHostAlloc` opened it
    itself. A buffer installed by :func:`open_device` (or a test's
    ``_set_udmabuf``) belongs to its installer, who may still be holding live
    ``HostRegion`` views into it, so this drops the reference without closing.
    Any :class:`~runtime.backend.HostRegion` carved from a buffer this
    call DOES close is dead afterwards: its view points at an unmapped page.
    """
    _set_udmabuf(None)


def minitpuHostRewind() -> None:
    """Reset the bump offset, KEEPING the udmabuf mapped.

    :func:`minitpuHostReset` drops the buffer as well, so the next allocation
    re-opens and re-mmaps it -- sixteen megabytes of it. A forward pass calls this
    once per layer to bound the allocator, and profiled on a board that re-mapping
    was 1.875 ms a call, 3.7 ms a layer, larger than either of the two host-to-
    device copies the layer makes.

    The distinction matters for correctness, not only speed, and in the harder
    direction: after a reset an old :class:`~runtime.backend.HostRegion` points at
    an unmapped page, which faults. After a rewind it points at memory a later
    allocation will hand out again, which does not. **Any region carved before this
    call is aliased afterwards** -- caching one across it is the mistake that took a
    board from 16/16 to 0/16. Use this only where the regions are consumed
    immediately, as every staging path in `gpt2_kernels` does.
    """
    global _BUMP_OFFSET
    _BUMP_OFFSET = 0


def _round_up(value: int, align: int) -> int:
    """Round ``value`` up to the next multiple of ``align``."""
    return (value + align - 1) // align * align


# ===========================================================================
# RFC P1 (kernarg arg-mode) — device (DMEM) allocator.
#
# A minimal first-fit / bump free-list over the DMEM aperture, returning
# opaque device pointers that arg-mode kernels bind at launch
# (``minitpuLaunchKernel(..., args=[ptr, ...])``).  This is the DEVICE pool
# (DMEM), distinct from :func:`minitpuHostAlloc` (the pinned-HOST udmabuf
# pool). The configured board address space provides the kernel-visible
# aperture; its low reserved prefix is used for kernel-image staging. Alignment is CDMA_ALIGN
# (32 B, the 256-bit AXI CDMA unit).  Opaque handle == physical base (no device
# MMU; §7 trust boundary: no HW bounds check, so the allocator MUST stay
# in-aperture and pointers passed to a kernel MUST be validated in-aperture).
#
# NOTE (Stage-2): this cannot be exercised by the tile cocotb DV (there is no
# board / MMIO there); it is board-facing host code, minimal and correct.
# ===========================================================================
# (base, size) free spans and base -> size allocations; a single global pool
# (mirrors the whole-device model; per-process, not multi-device).
_dev_pool: tuple[int, int] | None = None
_dev_free: list[tuple[int, int]] = []
_dev_alloc: dict[int, int] = {}


def _device_pool() -> tuple[int, int]:
    """Return the configured board's kernel-visible data range.

    The address space's low reserved prefix belongs to instruction-image
    staging. Deriving the remainder avoids using a ZCU102 physical-DDR address
    on boards such as the ZCU104, whose kernel aperture differs.
    """
    space = minitpuGetAddressSpace()
    return space.kernel_base + space.reserved, space.kernel_size - space.reserved


def _ensure_device_pool() -> None:
    global _dev_pool
    pool = _device_pool()
    if _dev_pool == pool:
        return
    if _dev_alloc:
        raise RuntimeError("cannot change the device address space while allocations are live")
    _dev_pool = pool
    _dev_free[:] = [pool]


def minitpuMalloc(nbytes: int) -> int:
    """Allocate ``nbytes`` of device (DMEM) memory; return a 32B-aligned pointer.

    First-fit over the free-list, size rounded up to :data:`CDMA_ALIGN`.  The
    returned opaque device pointer is the physical DMEM byte address, ready to
    pass to :func:`~runtime.launch.minitpuLaunchKernel` as an arg-mode
    param (base pointer).  Raises :class:`MemoryError` when the aperture cannot
    satisfy the request.
    """
    if nbytes <= 0:
        raise ValueError(f"minitpuMalloc: nbytes must be > 0, got {nbytes}")
    _ensure_device_pool()
    n = _round_up(int(nbytes), CDMA_ALIGN)
    for i, (base, size) in enumerate(_dev_free):
        if size >= n:
            if size == n:
                _dev_free.pop(i)
            else:
                _dev_free[i] = (base + n, size - n)
            _dev_alloc[base] = n
            return base
    raise MemoryError(
        f"DMEM aperture exhausted: need {n} bytes, largest free span is "
        f"{max((s for _, s in _dev_free), default=0)} bytes"
    )


def minitpuFree(addr: int) -> None:
    """Return a :func:`minitpuMalloc` device pointer's span to the free-list.

    Coalesces adjacent free spans so a subsequent malloc can reuse a merged
    region (avoids fragmenting the aperture across alloc/free cycles).  Raises
    :class:`KeyError` on a pointer this allocator did not hand out.
    """
    if addr not in _dev_alloc:
        raise KeyError(f"minitpuFree: 0x{addr:x} was not returned by minitpuMalloc")
    size = _dev_alloc.pop(addr)
    _dev_free.append((addr, size))
    _dev_free.sort()
    merged: list[tuple[int, int]] = []
    for base, sz in _dev_free:
        if merged and merged[-1][0] + merged[-1][1] == base:
            merged[-1] = (merged[-1][0], merged[-1][1] + sz)
        else:
            merged.append((base, sz))
    _dev_free[:] = merged


def minitpuDeviceReset() -> None:
    """Reset the device (DMEM) allocator to a single free aperture span.

    Test / re-run isolation, mirroring :func:`minitpuHostReset` for the host
    pool.  Drops every outstanding allocation without touching device memory.
    """
    global _dev_pool
    _dev_pool = _device_pool()
    _dev_free[:] = [_dev_pool]
    _dev_alloc.clear()


def minitpuInAperture(addr: int, nbytes: int = 0) -> bool:
    """True iff ``[addr, addr+nbytes)`` lies inside the DMEM aperture.

    The device has NO hardware bounds check (RFC §7 trust boundary), so the
    runtime SHOULD validate any pointer bound as a kernel arg with this before
    launch; a bad pointer walks off the buffer silently on hardware.
    """
    base, size = _device_pool()
    return base <= addr and (addr + max(0, nbytes)) <= (base + size)
