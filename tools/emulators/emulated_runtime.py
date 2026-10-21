"""Provides the board runtime's host API over the functional emulator, so the board's own code runs on a host.

The functions mirror board_package/runtime's: DMEM is one byte array the emulator's DRAM shares, an allocation is a
bump in it, a copy moves bytes between it and a host buffer, and a launch runs the image resident at its address.
Installing this module as `runtime` before board_package/gpt2_inference.py is imported runs every launch that module
makes, and every board check built on it, through the emulator.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to this file's directory, which holds the emulator."""

sys.path.insert(0, str(_HERE))

from minitpu_emulator import Machine, read_image  # noqa: E402

_DMEM_BYTES: int = 1 << 30
"""Bytes of emulated DMEM, enough for GPT-2 small's resident weights and the tensors one run allocates."""

_ALIGNMENT: int = 64
"""Byte alignment of every allocation, being the AXI CDMA's alignment on the board."""

_DMEM: NDArray[np.uint8] = np.zeros(_DMEM_BYTES, dtype=np.uint8)
"""Emulated DMEM, which every launch's DRAM shares."""

_NEXT: list[int] = [_ALIGNMENT]
"""The next free DMEM byte, starting past address zero so a null pointer never names an allocation."""

LAUNCHES: list[int] = [0]
"""Launches run so far."""

BUNDLES: list[int] = [0]
"""Bundles every launch has issued so far."""


class MemcpyKind:
    """Defines the direction a copy takes, as the board runtime does."""

    HostToDevice = "HostToDevice"
    """Copies from the host buffer to the device address."""
    DeviceToHost = "DeviceToHost"
    """Copies from the device address to the host buffer."""


class HostBuffer:
    """Holds one host buffer, exposing its words as the board runtime's HostRegion does.

    Args:
        nbytes: Bytes the buffer holds.

    Attributes:
        view: The buffer's contents as 32-bit words.
        phys_addr: A placeholder physical address, which the emulator never reads.
    """

    def __init__(self, nbytes: int) -> None:
        self.view = np.zeros(-(-int(nbytes) // 4), dtype=np.uint32)
        self.phys_addr = 0

    def sync_to_device(self) -> None:
        """Accepts a cache flush, which the emulator's host buffers, holding no cache, do not need."""


def minitpuSetDevice(uio_name: str) -> None:  # noqa: N802 -- mirrors the board runtime's name
    """Accepts a device name, of which the emulator has one."""


def minitpuHostReset() -> None:  # noqa: N802
    """Accepts a host reset, which the emulator's host buffers do not need."""


def minitpuHostAlloc(nbytes: int, **_: object) -> HostBuffer:  # noqa: N802
    """Returns a zeroed host buffer of `nbytes` bytes."""
    return HostBuffer(nbytes)


def minitpuMalloc(nbytes: int) -> int:  # noqa: N802
    """Returns the DMEM address of `nbytes` fresh bytes."""
    address = _NEXT[0]
    _NEXT[0] = address + -(-int(nbytes) // _ALIGNMENT) * _ALIGNMENT
    if _NEXT[0] > _DMEM_BYTES:
        message = f"Unable to allocate {nbytes} bytes. Emulated DMEM holds {_DMEM_BYTES} and is exhausted."
        raise MemoryError(message)
    return address


def minitpuFree(address: int) -> None:  # noqa: N802
    """Accepts a free, which the bump allocator does not reclaim."""


def minitpuMemcpy(dst: HostBuffer | int, src: HostBuffer | int, nbytes: int, kind: str) -> None:  # noqa: N802
    """Copies `nbytes` bytes between a host buffer and DMEM in the direction `kind` names."""
    if kind == MemcpyKind.HostToDevice:
        _DMEM[int(dst):int(dst) + nbytes] = src.view.view(np.uint8)[:nbytes]
    elif kind == MemcpyKind.DeviceToHost:
        dst.view.view(np.uint8)[:nbytes] = _DMEM[int(src):int(src) + nbytes]
    else:
        message = f"Unable to copy. The kind must be a MemcpyKind, but got {kind!r}."
        raise ValueError(message)


def minitpuLaunchKernel(kbin_addr: int, kbin_size: int, *, args: list[int] | None = None) -> None:  # noqa: N802
    """Runs the image resident at `kbin_addr` through the emulator with the given launch arguments."""
    image = _DMEM[kbin_addr:kbin_addr + kbin_size].tobytes()
    words = [int.from_bytes(image[offset:offset + 16], "little") for offset in range(0, len(image), 16)]
    machine = Machine(_DMEM, arguments=tuple(int(argument) & 0xFFFF_FFFF for argument in (args or [])))
    machine.run(words)
    LAUNCHES[0] += 1
    BUNDLES[0] += machine.bundles


def minitpuGetAddressSpace() -> None:  # noqa: N802
    """Returns no address space, which the emulator's flat DMEM does not have."""
    return None


__all__ = ["HostBuffer", "MemcpyKind", "minitpuFree", "minitpuGetAddressSpace", "minitpuHostAlloc",
           "minitpuHostReset", "minitpuLaunchKernel", "minitpuMalloc", "minitpuMemcpy", "minitpuSetDevice",
           "read_image"]
