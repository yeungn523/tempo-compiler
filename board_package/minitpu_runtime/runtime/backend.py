"""MMIO backend + host-buffer allocator for the minitpu host runtime.

The runtime never talks to hardware directly: every CMD-window write, doorbell
ring, and status poll goes through an :class:`Mmio` backend.  This keeps the
descriptor / doorbell / poll logic (``memcpy.py``, ``launch.py``) testable with
a :class:`MockMmio` and no board present, while a real board plugs in
:class:`UioMmio`.

v1 board HAL is raw Linux UIO + udmabuf (PYNQ is retired):

  - :class:`Mmio`           - abstract 32-bit register access (``read32`` / ``write32``).
  - :class:`MockMmio`       - records every write, replays scripted reads; ``instr_ready``
                              reports idle after ``ready_after`` polls so a memcpy/launch
                              completes deterministically in a unit test.
  - :class:`UioMmio`        - the MINITPU AXI-Lite slave window mmap'd through a Linux **UIO**
                              character device (``/dev/uioN``).  The device is discovered
                              by name via ``/sys/class/uio/uio*/name``.
  - :class:`UdmabufBuffer`  - a contiguous host DMA buffer backed by **udmabuf**
                              (``/dev/udmabufN``): its physical address (from sysfs) is the
                              ``host_phys`` operand of a COPY, and its cache is synced around
                              each transfer for H2D/D2H coherence.

Both device-backed classes import cleanly on a dev host: only *opening* a device
(``UioMmio.open_by_name`` / ``UdmabufBuffer.open``) touches ``/dev`` or ``/sys``,
and raises a clear :class:`RuntimeError` off-board.
"""

from __future__ import annotations

import mmap
import os
import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from minitpu.abi import AXI_REG

# ---------------------------------------------------------------------------
# sysfs / dev conventions (single place, no magic paths sprinkled around)
# ---------------------------------------------------------------------------
_UIO_CLASS = Path("/sys/class/uio")
_UDMABUF_CLASS = Path("/sys/class/u-dma-buf")
_REG_WORD = struct.Struct("<I")  # little-endian u32 register word


class Mmio(ABC):
    """Abstract 32-bit MMIO register access for the CMD window + status words.

    Offsets are AXI-Lite byte offsets from ``AXI_REG`` (0x00..0x24).  Both
    methods operate on a single 32-bit word.
    """

    @abstractmethod
    def read32(self, offset: int) -> int:
        """Read the 32-bit word at ``offset`` (AXI-Lite byte offset)."""

    @abstractmethod
    def write32(self, offset: int, value: int) -> None:
        """Write ``value`` (masked to 32 bits) to the word at ``offset``."""


class MockMmio(Mmio):
    """In-memory :class:`Mmio` for contract tests (no hardware).

    Records the full ordered write log and the last value per offset, replays
    scripted read values, and drives ``instr_ready`` to 1 after ``ready_after``
    polls so a synchronous memcpy/launch terminates.

    Attributes:
        writes:  ordered list of ``(offset, value)`` writes.
        written: last value written per offset.
        reads:   ordered list of read offsets.
    """

    def __init__(self, ready_after: int = 0, read_values: dict[int, int] | None = None):
        self.writes: list[tuple[int, int]] = []
        self.written: dict[int, int] = {}
        self.reads: list[int] = []
        self._ready_after = ready_after
        self._ready_polls = 0
        self._read_values: dict[int, int] = dict(read_values or {})

    def write32(self, offset: int, value: int) -> None:
        v = value & 0xFFFF_FFFF
        self.writes.append((offset, v))
        self.written[offset] = v

    def read32(self, offset: int) -> int:
        self.reads.append(offset)
        if offset == AXI_REG["instr_ready"]:
            self._ready_polls += 1
            return 1 if self._ready_polls > self._ready_after else 0
        return self._read_values.get(offset, 0)


# ---------------------------------------------------------------------------
# Register access over Linux UIO
# ---------------------------------------------------------------------------
class UioMmio(Mmio):
    """:class:`Mmio` backed by an mmap of the MINITPU AXI-Lite window via Linux UIO.

    The UIO framework exposes each mapped memory region of a device as an
    mmap-able window on ``/dev/uioN``.  We mmap map ``map_index`` (the MINITPU
    AXI-Lite slave, map0 by default) and do 32-bit little-endian register
    accesses straight into that mapping.

    Constructed from any writable buffer (an ``mmap`` on a board; a temp-file
    ``mmap`` in tests), so the register path is exercised without hardware.
    Use :meth:`open_by_name` on a board to discover and map the real device.
    """

    def __init__(self, mem, *, _owns: tuple = ()):
        # mem is any object supporting the writable buffer protocol at byte
        # offsets (mmap, bytearray, memoryview).
        self._mem = mem
        self._owns = _owns  # (fd, mmap) to release on close(), if we opened them

    def read32(self, offset: int) -> int:
        return _REG_WORD.unpack_from(self._mem, offset)[0]

    def write32(self, offset: int, value: int) -> None:
        _REG_WORD.pack_into(self._mem, offset, value & 0xFFFF_FFFF)

    def close(self) -> None:
        for obj in self._owns:
            try:
                obj.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    @classmethod
    def open_by_name(cls, name: str, *, map_index: int = 0) -> UioMmio:
        """Discover ``/dev/uioN`` whose ``.../name`` equals ``name`` and mmap it.

        Raises :class:`RuntimeError` when UIO is unavailable (off-board) or no
        device with that name exists.  The map size comes from the UIO sysfs
        ``maps/map{map_index}/size`` node; the map is selected by the standard
        UIO mmap offset convention (``offset = map_index * PAGE_SIZE``).
        """
        cached = _UIO_CACHE.get((name, map_index))
        if cached is not None:
            return cached
        uio = _find_uio_by_name(name)
        size = _uio_map_size(uio, map_index)
        dev = f"/dev/{uio}"
        try:
            fd = os.open(dev, os.O_RDWR | os.O_SYNC)
        except OSError as exc:  # pragma: no cover - board-only path
            raise RuntimeError(f"cannot open UIO device {dev!r}: {exc}") from exc
        mem = mmap.mmap(fd, size, offset=map_index * mmap.PAGESIZE)
        os.close(fd)  # mmap keeps the mapping alive after the fd is closed
        opened = cls(mem, _owns=(mem,))
        _UIO_CACHE[(name, map_index)] = opened
        return opened


#: Mappings already opened, keyed by (uio name, map index).
#:
#: `minitpuSetDevice` calls `open_by_name`, and the drivers call it once per
#: launch -- seventeen times in one GPT-2 layer. Each call re-globbed
#: /sys/class/uio, read every name node, and made a FRESH mmap of the same
#: window: measured on a board, that path was 106 file opens and 121 stats per
#: layer, and the mappings were never released. A UIO node's index and size do
#: not change while a process runs, so this is cached. `reset_uio_cache` drops it
#: for a process that has to rediscover, and `open_device` calls that.
_UIO_CACHE: dict[tuple[str, int], "UioMmio"] = {}


def reset_uio_cache() -> None:
    """Forgets every cached UIO mapping, closing the ones this module owns."""
    for mmio in _UIO_CACHE.values():
        try:
            mmio.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    _UIO_CACHE.clear()


def _find_uio_by_name(name: str) -> str:
    """Return the ``uioN`` directory name whose UIO ``name`` node equals ``name``."""
    if not _UIO_CLASS.is_dir():  # pragma: no cover - board-only path
        raise RuntimeError(
            f"UIO not available ({_UIO_CLASS} missing); this backend needs a "
            "minitpu board with the UIO driver bound to the MINITPU AXI-Lite slave"
        )
    seen: list[str] = []
    for entry in sorted(_UIO_CLASS.glob("uio*")):
        name_node = entry / "name"
        if not name_node.exists():
            continue
        this = name_node.read_text().strip()
        seen.append(f"{entry.name}={this!r}")
        if this == name:
            return entry.name
    raise RuntimeError(f"no UIO device named {name!r} under {_UIO_CLASS} (found: {seen or 'none'})")


def _uio_map_size(uio: str, map_index: int) -> int:
    """Read the byte size of UIO memory map ``map_index`` from sysfs."""
    size_node = _UIO_CLASS / uio / "maps" / f"map{map_index}" / "size"
    try:
        return int(size_node.read_text().strip(), 0)
    except OSError as exc:  # pragma: no cover - board-only path
        raise RuntimeError(f"cannot read UIO map size {size_node!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Host DMA buffers over udmabuf
# ---------------------------------------------------------------------------
class UdmabufBuffer:
    """A contiguous, physically-addressed host DMA buffer backed by udmabuf.

    Satisfies the ``HostBuffer`` protocol (``phys_addr`` + ``sync_to_device`` /
    ``sync_from_device``) that :func:`runtime.memcpy.minitpuMemcpy`
    consumes.  ``phys_addr`` (from the udmabuf sysfs ``phys_addr`` node) is the
    COPY descriptor's ``host_phys``; ``mem`` is the CPU-side mmap of the buffer.

    The mapping is CACHEABLE by default (see :meth:`open`).  The PS-PL HP port
    is non-coherent, so callers MUST bracket every transfer with the sysfs sync
    nodes to stay correct:
      - :meth:`sync_to_device`   flushes CPU cache -> DDR before an H2D transfer
        (write "1" to ``sync_for_device``), so the copy engine reads fresh data.
      - :meth:`sync_from_device` invalidates CPU cache <- DDR after a D2H
        transfer (write "1" to ``sync_for_cpu``), so the CPU sees device writes.

    The sync writes fire whenever the sysfs node is present (the cacheable
    udmabuf exposes them); they are skipped when absent (e.g. a non-cached
    instance where they would be a no-op).

    Both sync calls take a byte range, because cache maintenance costs time in
    proportion to the range it sweeps.  Omitting the range sweeps the whole
    buffer.  On this board the sweep is the cheap part: 73 us per MiB, so 0.6 us
    for the 9216 bytes a launch moves.  What a sync actually costs is the sysfs
    writes that carry it, so the descriptors below are opened once and written
    with :func:`os.pwrite`.  Measured on the ZCU104, one write costs 238.9 us
    through ``Path.write_text``, 33.5 us reopening the node each time, and 5.7 us
    on a held descriptor -- the node's store handler is 5.7 us of that and the
    rest is the interpreter building a file object per write.
    """

    _SYNC_NODES: tuple[str, ...] = ("sync_for_device", "sync_for_cpu", "sync_offset", "sync_size")
    """Sysfs nodes held open for the buffer's lifetime, being every node a sync writes."""

    def __init__(self, mem, phys_addr: int, size: int, *, sysfs_dir=None, _owns: tuple = ()):
        self.mem = mem
        self.phys_addr = int(phys_addr)
        self.size = int(size)
        self._sysfs = Path(sysfs_dir) if sysfs_dir is not None else None
        self._owns = _owns
        self._nodes: dict[str, int] = {}
        self._window: tuple[int, int] | None = None
        self._open_sync_nodes()

    def _open_sync_nodes(self) -> None:
        """Holds a write descriptor for every sync node this instance exposes.

        Opening once moves the node's existence and writability check out of the sync path with it: a node absent
        or read-only simply has no descriptor, which is the same condition the per-call checks used to report.
        """
        if self._sysfs is None:
            return
        for name in self._SYNC_NODES:
            try:
                self._nodes[name] = os.open(self._sysfs / name, os.O_WRONLY)
            except OSError:  # absent, or not writable by this user -- leave the driver's default in force
                continue

    def _write_node(self, name: str, value: int) -> None:
        """Writes one integer to a held sync descriptor."""
        os.pwrite(self._nodes[name], str(value).encode(), 0)

    def sync_to_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        """Flushes CPU cache to DDR before the device reads this buffer (H2D).

        Args:
            offset: The byte offset of the range to sync within this buffer.
            nbytes: The length of that range, defaulting to the rest of the buffer.
        """
        self._sync("sync_for_device", offset, nbytes)

    def sync_from_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        """Invalidates CPU cache after the device wrote this buffer (D2H).

        Args:
            offset: The byte offset of the range to sync within this buffer.
            nbytes: The length of that range, defaulting to the rest of the buffer.
        """
        self._sync("sync_for_cpu", offset, nbytes)

    def _sync(self, node: str, offset: int = 0, nbytes: int | None = None) -> None:
        if node not in self._nodes:
            return
        self._select_window(offset, self.size - offset if nbytes is None else nbytes)
        self._write_node(node, 1)

    def _select_window(self, offset: int, nbytes: int) -> None:
        """Narrows the driver's next sync to ``[offset, offset + nbytes)``.

        The ``sync_offset`` and ``sync_size`` nodes hold sticky state that the next ``sync_for_cpu`` or
        ``sync_for_device`` write consumes.  Being sticky, a window equal to the one already in force needs no
        write at all, which is what the repeated sync of one region costs nothing to skip.  Both nodes are written
        or neither is, since writing one alone would leave a range neither the caller nor the driver default asked
        for.  An instance whose nodes are absent or read-only has no descriptor for them and keeps the driver's
        whole-buffer default.

        Args:
            offset: The byte offset of the range within this buffer.
            nbytes: The length of the range in bytes.
        """
        if not {"sync_offset", "sync_size"} <= self._nodes.keys():
            return
        if self._window == (offset, nbytes):
            return
        self._write_node("sync_offset", offset)
        self._write_node("sync_size", nbytes)
        self._window = (offset, nbytes)

    def close(self) -> None:
        for descriptor in self._nodes.values():
            try:
                os.close(descriptor)
            except OSError:  # noqa: PERF203 - best-effort teardown
                pass
        self._nodes = {}
        self._window = None
        for obj in self._owns:
            try:
                obj.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    @classmethod
    def open(
        cls,
        index: int = 0,
        *,
        size: int | None = None,
        sysfs_dir=None,
        dev_path=None,
        cached: bool = True,
    ) -> UdmabufBuffer:
        """Open udmabuf device ``index`` and mmap it for host access.

        Reads the buffer's physical address and size from the udmabuf sysfs
        directory (``/sys/class/u-dma-buf/udmabuf{index}`` by default), then
        mmaps ``/dev/udmabuf{index}``.
        ``sysfs_dir`` / ``dev_path`` / ``size`` override the discovery for tests.

        The mapping is CACHEABLE by default (``cached=True``): the device is
        opened WITHOUT ``O_SYNC``, giving Normal (cacheable) memory.  Callers
        MUST then bracket every transfer with :meth:`sync_to_device` (flush
        before the fabric reads) and :meth:`sync_from_device` (invalidate after
        the fabric writes), because the PS-PL HP port is non-coherent.  This is
        the standard streaming-DMA contract (Linux ``dma_sync_*``, PYNQ
        ``allocate(cacheable=True)`` + flush/invalidate, XRT ``xrt::bo`` sync).

        ``cached=False`` restores the old non-cached (``O_SYNC``) mapping, which
        is ARM64 Device-type memory.  That is UNSAFE for numpy: glibc's AArch64
        ``memset`` uses ``DC ZVA`` for bulk zeroing at >=256 bytes, which is
        illegal on Device memory and faults with SIGBUS (reproduced on zcu102),
        and unaligned / SIMD accesses are likewise illegal.  Kept only as an
        escape hatch; do not use it for numpy-backed buffers.

        Raises :class:`RuntimeError` when udmabuf is unavailable (off-board).
        """
        name = f"udmabuf{index}"
        sysfs = Path(sysfs_dir) if sysfs_dir is not None else _udmabuf_sysfs(name)
        phys = _read_udmabuf_phys(sysfs)
        if size is None:
            size = int((sysfs / "size").read_text().strip(), 0)
        dev = str(dev_path) if dev_path is not None else f"/dev/{name}"
        flags = os.O_RDWR if cached else os.O_RDWR | os.O_SYNC
        try:
            fd = os.open(dev, flags)
        except OSError as exc:  # pragma: no cover - board-only path
            raise RuntimeError(f"cannot open udmabuf device {dev!r}: {exc}") from exc
        mem = mmap.mmap(fd, size)
        os.close(fd)
        return cls(mem, phys, size, sysfs_dir=sysfs, _owns=(mem,))


def _udmabuf_sysfs(name: str) -> Path:
    """Locate the udmabuf sysfs directory for ``name`` (class dir or device tree)."""
    class_dir = _UDMABUF_CLASS / name
    if (class_dir / "phys_addr").exists():
        return class_dir
    # Fallback: some kernels expose it only under /sys/devices/.../<name>/.
    for cand in Path("/sys/devices").rglob(name):
        if (cand / "phys_addr").exists():
            return cand
    raise RuntimeError(
        f"udmabuf {name!r} not found (looked in {_UDMABUF_CLASS} and /sys/devices); "
        "this backend needs a minitpu board with a udmabuf instance allocated"
    )


def _read_udmabuf_phys(sysfs_dir: Path) -> int:
    """Read the udmabuf physical base address (hex) from its sysfs ``phys_addr``."""
    node = Path(sysfs_dir) / "phys_addr"
    try:
        return int(node.read_text().strip(), 16)
    except OSError as exc:  # pragma: no cover - board-only path
        raise RuntimeError(f"cannot read udmabuf phys_addr {node!r}: {exc}") from exc


class HostRegion:
    """A phys-addressed sub-view of one contiguous :class:`UdmabufBuffer`.

    A generic, board-AGNOSTIC carve-out: given a parent udmabuf, a byte offset,
    and a word count, it exposes an integer ``phys_addr`` (parent base + offset)
    and a numpy uint32 ``view`` over that slice of the parent's mmap.  It carries
    NO board constant; the offset/size are supplied by the caller.

    It satisfies the ``HostBuffer`` protocol consumed by
    :func:`runtime.memcpy.minitpuMemcpy` (an integer ``phys_addr`` plus
    optional ``sync_to_device`` / ``sync_from_device`` cache hooks), so a region
    is a valid COPY host operand.  Cache sync delegates to the parent naming this
    region's own byte range, since a region is the unit the caller transfers.  The
    allocator that hands out regions keeps them cache-line aligned, so a region's
    edge cache lines hold only its own data.
    """

    def __init__(self, parent: UdmabufBuffer, byte_off: int, words: int):
        self._parent = parent
        self._byte_off = int(byte_off)
        self._nbytes = int(words) * 4
        self.phys_addr = parent.phys_addr + byte_off  # COPY host_phys operand
        self.view = np.frombuffer(parent.mem, dtype=np.uint32, count=words, offset=byte_off)

    def sync_to_device(self) -> None:
        """Flushes this region's bytes out of the CPU cache (H2D)."""
        self._parent.sync_to_device(self._byte_off, self._nbytes)

    def sync_from_device(self) -> None:
        """Invalidates this region's bytes in the CPU cache (D2H)."""
        self._parent.sync_from_device(self._byte_off, self._nbytes)
