"""Provides the MMIO register backend and the host DMA buffer allocator the runtime drives a board through.

Every register access goes through an ``Mmio``, so a command sequence also runs against a mock with no board.
"""

from __future__ import annotations

import mmap
import os
import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from minitpu.abi import AXI_REG

_UIO_CLASS: Path = Path("/sys/class/uio")
"""Sysfs directory that lists every UIO device."""

_UDMABUF_CLASS: Path = Path("/sys/class/u-dma-buf")
"""Sysfs directory that lists every udmabuf instance."""

_REG_WORD: struct.Struct = struct.Struct("<I")
"""Little-endian layout of one 32-bit register word."""

_UIO_CACHE: dict[tuple[str, int], UioMmio] = {}
"""Mappings already opened, keyed by UIO node name and map index.

Discovery re-globbed sysfs and re-mmapped the same window once per launch, measured at 106 file opens per GPT-2
layer with nothing released. A node's index and size hold still while a process runs, which makes caching safe.
"""


class Mmio(ABC):
    """Defines 32-bit register access to the CMD window and its status words.

    Offsets are AXI-Lite byte offsets from ``AXI_REG``.
    """

    @abstractmethod
    def read32(self, offset: int) -> int:
        """Reads the register word at the given byte offset."""

    @abstractmethod
    def write32(self, offset: int, value: int) -> None:
        """Writes a value, masked to 32 bits, to the register word at the given byte offset."""


class MockMmio(Mmio):
    """Records register traffic in memory, so a command sequence runs with no hardware present.

    Args:
        ready_after: The number of busy ``instr_ready`` polls to replay before the flag reports idle, which is
            what lets a synchronous memcpy or launch terminate.
        read_values: The value each offset replays on a read, defaulting to zero.

    Attributes:
        writes: Ordered log of every write as an offset and value pair.
        written: Last value written per offset.
        reads: Ordered log of every read offset.
        _ready_after: Cached number of busy polls to replay.
        _ready_polls: Count of ``instr_ready`` polls served so far.
        _read_values: Cached scripted read values.
    """

    def __init__(self, ready_after: int = 0, read_values: dict[int, int] | None = None):
        self.writes: list[tuple[int, int]] = []
        self.written: dict[int, int] = {}
        self.reads: list[int] = []
        self._ready_after = ready_after
        self._ready_polls = 0
        self._read_values: dict[int, int] = dict(read_values or {})

    def write32(self, offset: int, value: int) -> None:
        """Logs a masked write against the given offset."""
        masked = value & 0xFFFF_FFFF
        self.writes.append((offset, masked))
        self.written[offset] = masked

    def read32(self, offset: int) -> int:
        """Logs a read and replays the scripted value, driving ``instr_ready`` from the poll count."""
        self.reads.append(offset)
        if offset == AXI_REG["instr_ready"]:
            self._ready_polls += 1
            return 1 if self._ready_polls > self._ready_after else 0
        return self._read_values.get(offset, 0)


class UioMmio(Mmio):
    """Serves register access from an mmap of the MINITPU AXI-Lite window exposed through Linux UIO.

    Any writable buffer is accepted, so the register path runs against a temporary file off-board.

    Args:
        mem: A writable buffer addressed at byte offsets, being an mmap on a board.
        _owns: The objects to close when this instance is closed.

    Attributes:
        _mem: Cached writable buffer holding the register window.
        _owns: Cached objects this instance closes.
    """

    def __init__(self, mem, *, _owns: tuple = ()):
        self._mem = mem
        self._owns = _owns

    def read32(self, offset: int) -> int:
        """Reads the register word at the given byte offset."""
        return _REG_WORD.unpack_from(self._mem, offset)[0]

    def write32(self, offset: int, value: int) -> None:
        """Writes a masked value to the register word at the given byte offset."""
        _REG_WORD.pack_into(self._mem, offset, value & 0xFFFF_FFFF)

    def close(self) -> None:
        """Releases the mapping this instance opened."""
        for owned in self._owns:
            try:
                owned.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    @classmethod
    def open_by_name(cls, name: str, *, map_index: int = 0) -> UioMmio:
        """Discovers the /dev/uioN node carrying the given UIO name and mmaps one of its memory maps.

        The map size comes from sysfs, and the map is selected by the UIO convention that multiplies the map index
        by the page size to form the mmap offset. Repeated calls for one node and map index return the cache.

        Args:
            name: The UIO name node contents that identify the device.
            map_index: The index of the device's memory map to mmap.

        Raises:
            RuntimeError: When UIO is unavailable off-board, no device carries that name, or the node cannot be
                opened.
        """
        cached = _UIO_CACHE.get((name, map_index))
        if cached is not None:
            return cached
        uio = _find_uio_by_name(name)
        size = _uio_map_size(uio, map_index)
        device = f"/dev/{uio}"
        try:
            descriptor = os.open(device, os.O_RDWR | os.O_SYNC)
        except OSError as exc:  # pragma: no cover - board-only path
            message = f"Unable to open the UIO device {device!r}. The open failed with {exc}."
            raise RuntimeError(message) from exc
        mem = mmap.mmap(descriptor, size, offset=map_index * mmap.PAGESIZE)
        # Closes the descriptor, since the mmap holds the mapping alive on its own.
        os.close(descriptor)
        opened = cls(mem, _owns=(mem,))
        _UIO_CACHE[(name, map_index)] = opened
        return opened


class UdmabufBuffer:
    """Holds a contiguous, physically addressed host DMA buffer backed by udmabuf.

    The mapping is cacheable and the PS-PL HP port is not coherent, so a caller brackets every transfer with the
    sync calls. Both sync calls take a byte range, since cache maintenance costs 73 us per MiB here. The sysfs
    write carrying a sync dominates its cost, so the nodes are opened once and written through ``os.pwrite``:
    5.7 us on a held descriptor against 238.9 us through ``Path.write_text``.

    Args:
        mem: The CPU-side mmap of the buffer.
        phys_addr: The physical byte address of the buffer.
        size: The byte size of the buffer.
        sysfs_dir: The udmabuf sysfs directory exposing the sync nodes.
        _owns: The objects to close when this instance is closed.

    Attributes:
        mem: The CPU-side mmap of the buffer.
        phys_addr: The physical byte address a COPY descriptor carries.
        size: The byte size of the buffer.
        _sysfs: Cached path to the udmabuf sysfs directory.
        _owns: Cached objects this instance closes.
        _nodes: Write descriptors held open for each sync node the instance exposes.
        _window: Byte range currently latched into the driver's sync nodes.
        _unwritable: Sync nodes that exist but that this account cannot write.
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
        self._unwritable: list[str] = []
        self._open_sync_nodes()

    def sync_to_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        """Flushes the CPU cache to DDR before the device reads the given range, defaulting to the whole buffer."""
        self._sync("sync_for_device", offset, nbytes)

    def sync_from_device(self, offset: int = 0, nbytes: int | None = None) -> None:
        """Invalidates the CPU cache after the device wrote the given range, defaulting to the whole buffer."""
        self._sync("sync_for_cpu", offset, nbytes)

    def close(self) -> None:
        """Releases the sync descriptors and the mapping this instance opened."""
        for descriptor in self._nodes.values():
            try:
                os.close(descriptor)
            except OSError:  # noqa: PERF203 - best-effort teardown
                pass
        self._nodes = {}
        self._window = None
        for owned in self._owns:
            try:
                owned.close()
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
        """Opens a udmabuf device and mmaps it for host access.

        An uncached mapping is ARM64 Device-type memory, where glibc's ``memset`` faults with SIGBUS because it
        zeroes with ``DC ZVA`` at 256 bytes and above, so it is an escape hatch for buffers numpy never touches.

        Args:
            index: The udmabuf instance to open.
            size: The byte size to map, defaulting to the size sysfs reports.
            sysfs_dir: The udmabuf sysfs directory, defaulting to the one discovery finds.
            dev_path: The character device to open, defaulting to the one named by the index.
            cached: Determines whether the mapping is cacheable.

        Raises:
            RuntimeError: When udmabuf is unavailable off-board or the device cannot be opened.
        """
        name = f"udmabuf{index}"
        sysfs = Path(sysfs_dir) if sysfs_dir is not None else _udmabuf_sysfs(name)
        phys = _read_udmabuf_phys(sysfs)
        if size is None:
            size = int((sysfs / "size").read_text().strip(), 0)
        device = str(dev_path) if dev_path is not None else f"/dev/{name}"
        flags = os.O_RDWR if cached else os.O_RDWR | os.O_SYNC
        try:
            descriptor = os.open(device, flags)
        except OSError as exc:  # pragma: no cover - board-only path
            message = f"Unable to open the udmabuf device {device!r}. The open failed with {exc}."
            raise RuntimeError(message) from exc
        mem = mmap.mmap(descriptor, size)
        os.close(descriptor)
        return cls(mem, phys, size, sysfs_dir=sysfs, _owns=(mem,))

    def _open_sync_nodes(self) -> None:
        """Holds a write descriptor for every sync node this instance exposes.

        Opening once takes the node's existence and writability check out of the sync path with it, so a node that
        is absent or read-only simply has no descriptor.
        """
        if self._sysfs is None:
            return
        for name in self._SYNC_NODES:
            try:
                self._nodes[name] = os.open(self._sysfs / name, os.O_WRONLY)
            except OSError:
                # A missing node leaves the driver's whole-buffer default in force, which is safe. A node that exists
                # and rejects the write holds sticky state another account last set, so that account's range is what
                # every sync consumes. The two are recorded apart and judged below.
                if (self._sysfs / name).exists():
                    self._unwritable.append(name)
        self._check_window_is_ours()

    def _check_window_is_ours(self) -> None:
        """Refuses a buffer whose sync window this account cannot set, naming the fix.

        Raises:
            RuntimeError: If a trigger is writable but the window nodes are present and are not.
        """
        triggers = {"sync_for_cpu", "sync_for_device"} & self._nodes.keys()
        stolen = [name for name in ("sync_offset", "sync_size") if name in self._unwritable]
        if not triggers or not stolen:
            return
        message = (
            f"Unable to own the sync window of {self._sysfs}. This account holds "
            f"{', '.join(sorted(triggers))} but cannot write {', '.join(stolen)}, which is the sticky byte range "
            f"every sync consumes. Another account latched that range, so a device-to-host copy would invalidate "
            f"their bytes instead of this buffer's and the host would read stale cache while the copy engine "
            f"reported success. Reload the driver so the nodes are recreated for this account:\n"
            f"  sudo rmmod u-dma-buf 2>/dev/null || sudo rmmod u_dma_buf\n"
            f"  sudo ./minitpu/scripts/provision.sh\n"
            f"Or open the buffer with cached=False, which maps it O_SYNC and needs no sync at all."
        )
        raise RuntimeError(message)

    def _write_node(self, name: str, value: int) -> None:
        """Writes one integer to a held sync descriptor."""
        os.pwrite(self._nodes[name], str(value).encode(), 0)

    def _sync(self, node: str, offset: int = 0, nbytes: int | None = None) -> None:
        """Narrows the sync window to the given range and writes the given sync node."""
        if node not in self._nodes:
            return
        self._select_window(offset, self.size - offset if nbytes is None else nbytes)
        self._write_node(node, 1)

    def _select_window(self, offset: int, nbytes: int) -> None:
        """Latches the given byte range into the driver's sticky offset and size nodes.

        A window equal to the one in force needs no write, so repeated syncs of one region cost nothing. Both
        nodes are written or neither is, since writing one alone leaves a range nobody asked for.
        """
        if not {"sync_offset", "sync_size"} <= self._nodes.keys():
            return
        if self._window == (offset, nbytes):
            return
        self._write_node("sync_offset", offset)
        self._write_node("sync_size", nbytes)
        self._window = (offset, nbytes)


class HostRegion:
    """Exposes a board-agnostic, physically addressed sub-view of one contiguous host DMA buffer.

    Cache sync delegates to the parent naming this region's own byte range, since a region is the unit the caller
    transfers. The allocator that hands out regions keeps them cache-line aligned, so a region's edge cache lines
    hold only its own data.

    Args:
        parent: The host DMA buffer this region is carved from.
        byte_off: The byte offset of the region within the parent.
        words: The length of the region in 32-bit words.

    Attributes:
        phys_addr: The physical byte address a COPY descriptor carries.
        view: A numpy uint32 view over the region's slice of the parent's mmap.
        _parent: Cached parent buffer that carries out cache maintenance.
        _byte_off: Cached byte offset of the region within the parent.
        _nbytes: Cached byte length of the region.
    """

    def __init__(self, parent: UdmabufBuffer, byte_off: int, words: int):
        self._parent = parent
        self._byte_off = int(byte_off)
        self._nbytes = int(words) * 4
        self.phys_addr = parent.phys_addr + byte_off
        self.view = np.frombuffer(parent.mem, dtype=np.uint32, count=words, offset=byte_off)

    def sync_to_device(self) -> None:
        """Flushes this region's bytes out of the CPU cache before a host to device transfer."""
        self._parent.sync_to_device(self._byte_off, self._nbytes)

    def sync_from_device(self) -> None:
        """Invalidates this region's bytes in the CPU cache after a device to host transfer."""
        self._parent.sync_from_device(self._byte_off, self._nbytes)


def reset_uio_cache() -> None:
    """Forgets every cached UIO mapping, closing the ones this module owns."""
    for mmio in _UIO_CACHE.values():
        try:
            mmio.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    _UIO_CACHE.clear()


def _find_uio_by_name(name: str) -> str:
    """Returns the uioN directory name whose UIO name node carries the given name.

    Raises:
        RuntimeError: When the UIO class directory is missing off-board, or no device carries that name.
    """
    if not _UIO_CLASS.is_dir():  # pragma: no cover - board-only path
        message = (
            f"Unable to find a UIO device. The class directory {_UIO_CLASS} is missing, so this backend needs a "
            f"minitpu board with the UIO driver bound to the MINITPU AXI-Lite slave."
        )
        raise RuntimeError(message)
    seen: list[str] = []
    for entry in sorted(_UIO_CLASS.glob("uio*")):
        name_node = entry / "name"
        if not name_node.exists():
            continue
        this = name_node.read_text().strip()
        seen.append(f"{entry.name}={this!r}")
        if this == name:
            return entry.name
    message = f"Unable to find a UIO device named {name!r} under {_UIO_CLASS}. Found {seen or 'none'}."
    raise RuntimeError(message)


def _uio_map_size(uio: str, map_index: int) -> int:
    """Reads the byte size of one UIO memory map from sysfs.

    Raises:
        RuntimeError: When the size node cannot be read.
    """
    size_node = _UIO_CLASS / uio / "maps" / f"map{map_index}" / "size"
    try:
        return int(size_node.read_text().strip(), 0)
    except OSError as exc:  # pragma: no cover - board-only path
        message = f"Unable to read the UIO map size {size_node!r}. The read failed with {exc}."
        raise RuntimeError(message) from exc


def _udmabuf_sysfs(name: str) -> Path:
    """Locates the udmabuf sysfs directory, in the class directory or under the device tree.

    Raises:
        RuntimeError: When neither location holds the instance.
    """
    class_dir = _UDMABUF_CLASS / name
    if (class_dir / "phys_addr").exists():
        return class_dir
    # Falls back to the device tree, since some kernels expose the instance only under /sys/devices.
    for candidate in Path("/sys/devices").rglob(name):
        if (candidate / "phys_addr").exists():
            return candidate
    message = (
        f"Unable to find the udmabuf {name!r} in {_UDMABUF_CLASS} or /sys/devices. This backend needs a minitpu "
        f"board with a udmabuf instance allocated."
    )
    raise RuntimeError(message)


def _read_udmabuf_phys(sysfs_dir: Path) -> int:
    """Reads the udmabuf physical base address from its sysfs node.

    Raises:
        RuntimeError: When the address node cannot be read.
    """
    node = Path(sysfs_dir) / "phys_addr"
    try:
        return int(node.read_text().strip(), 16)
    except OSError as exc:  # pragma: no cover - board-only path
        message = f"Unable to read the udmabuf phys_addr {node!r}. The read failed with {exc}."
        raise RuntimeError(message) from exc
