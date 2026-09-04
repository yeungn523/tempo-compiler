"""Board address mapping for kernel pointers and control-plane COPY traffic."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


def _contains(base: int, size: int, addr: int, nbytes: int) -> bool:
    return nbytes >= 0 and addr >= base and addr <= base + size and nbytes <= base + size - addr


@dataclass(frozen=True)
class DeviceAddressSpace:
    """Two AXI views of one DMEM aperture.

    Runtime device pointers use ``kernel_base``. COPY descriptors use
    ``copy_base``. Bytes beyond ``kernel_size`` are raw COPY storage only.

    ``kernel_alias_copy_base`` is the COPY (CDMA) address that ``kernel_base``
    actually refers to *in the DRAM itself*.  It defaults to ``copy_base`` --
    correct whenever both masters see the memory at the same offset (ZCU102) --
    but on a board where the two masters map the same DDR slave at different
    segment offsets it MUST be given explicitly.  See
    ``ZCU104_ADDRESS_SPACE`` below for why: AXI SmartConnect does NOT subtract
    the segment base, so two apertures at different offsets are two different
    regions of DRAM, not aliases of one.
    """

    copy_base: int
    copy_size: int
    kernel_base: int
    kernel_size: int
    reserved: int = 0x0010_0000
    kernel_alias_copy_base: int | None = None

    def __post_init__(self) -> None:
        if min(self.copy_base, self.copy_size, self.kernel_base, self.kernel_size) < 0:
            raise ValueError("address-space bases and sizes must be non-negative")
        if self.copy_size <= 0 or self.kernel_size <= 0:
            raise ValueError("address-space sizes must be positive")
        if self.kernel_size > self.copy_size:
            raise ValueError("kernel aperture cannot exceed the backing COPY aperture")
        if not (0 <= self.reserved < self.kernel_size):
            raise ValueError("reserved prefix must lie within the kernel aperture")
        if self.copy_base + self.copy_size > (1 << 64):
            raise ValueError("COPY aperture must fit an unsigned 64-bit address")
        if self.kernel_base + self.kernel_size > (1 << 32):
            raise ValueError("kernel aperture must fit the current 32-bit MINITPU address space")
        if not _contains(self.copy_base, self.copy_size, self.kernel_alias_base, self.kernel_size):
            raise ValueError(
                "the kernel aperture's COPY alias "
                f"[0x{self.kernel_alias_base:x}, "
                f"0x{self.kernel_alias_base + self.kernel_size:x}) must lie inside the "
                f"COPY aperture [0x{self.copy_base:x}, 0x{self.copy_base + self.copy_size:x})"
            )

    @property
    def kernel_alias_base(self) -> int:
        """COPY address of the first byte the MINITPU sees at ``kernel_base``."""
        return self.copy_base if self.kernel_alias_copy_base is None else self.kernel_alias_copy_base

    def contains_kernel(self, addr: int, nbytes: int = 0) -> bool:
        return _contains(self.kernel_base, self.kernel_size, addr, nbytes)

    def contains_copy(self, addr: int, nbytes: int = 0) -> bool:
        return _contains(self.copy_base, self.copy_size, addr, nbytes)

    def to_copy(self, addr: int, nbytes: int = 0) -> int:
        """Translate a kernel pointer; also accept a raw CDMA address."""
        if self.contains_kernel(addr, nbytes):
            return self.kernel_alias_base + addr - self.kernel_base
        if self.contains_copy(addr, nbytes):
            return addr
        raise ValueError(
            f"device range [0x{addr:x}, 0x{addr + nbytes:x}) is outside both "
            f"kernel [0x{self.kernel_base:x}, 0x{self.kernel_base + self.kernel_size:x}) "
            f"and COPY [0x{self.copy_base:x}, 0x{self.copy_base + self.copy_size:x}) apertures"
        )

    def to_kernel(self, addr: int, nbytes: int = 0) -> int:
        """Translate a CDMA address in the shared prefix to a kernel pointer."""
        if self.contains_kernel(addr, nbytes):
            return addr
        if self.contains_copy(addr, nbytes):
            offset = addr - self.kernel_alias_base
            if 0 <= offset and offset + nbytes <= self.kernel_size:
                return self.kernel_base + offset
        raise ValueError(
            f"COPY range [0x{addr:x}, 0x{addr + nbytes:x}) has no kernel-visible alias"
        )


ZCU102_ADDRESS_SPACE = DeviceAddressSpace(
    copy_base=0xA000_0000,
    copy_size=0x2000_0000,
    kernel_base=0xA000_0000,
    kernel_size=0x2000_0000,
)

# ZCU104: the CDMA (40-bit master) reaches the 16 GiB DMEM SODIMM through a
# segment based at 0x4_0000_0000; the MINITPU's 32-bit kernel-data master reaches
# the SAME DDR4 slave through a 2 GiB segment based at 0x8000_0000
# (fpga/boards/zcu104/ps.tcl, board_configure_minitpu_axi).
#
# These two segments are NOT aliases of the same DRAM bytes.  AXI SmartConnect
# decodes the segment to pick a route and then forwards the address truncated
# to the slave port's width (34 bits for this 16 GiB DDR4) -- it never subtracts
# the segment base.  So CDMA 0x4_0000_0000 lands on DDR byte 0, while MINITPU
# 0x8000_0000 lands on DDR byte 0x8000_0000, i.e. 2 GiB further in.  Measured
# directly against the generated SmartConnect netlist by
# fpga/microbench/smc_aperture_sim (see its README).
#
# The kernel aperture therefore aliases COPY address 0x4_8000_0000, not
# 0x4_0000_0000.  Getting this wrong is silent: the COPY completes cleanly, the
# LAUNCH loads whatever uninitialised DRAM sits 2 GiB away, and the tile
# executes garbage (opcode 0x00 = mxu.load.w, never HALT) until the command
# processor is stuck in CP_LAUNCH forever.  That was the real-hardware LAUNCH
# failure; see board_package/README.md.
ZCU104_ADDRESS_SPACE = DeviceAddressSpace(
    copy_base=0x4_0000_0000,
    copy_size=0x4_0000_0000,
    kernel_base=0x8000_0000,
    kernel_size=0x8000_0000,
    kernel_alias_copy_base=0x4_8000_0000,
)

# The constants above are a fallback.  The authoritative map for a given
# bitstream is emitted beside it as platform.json by fpga/build_bd_bitstream.tcl
# from the board's ps.tcl, so an image and its address map travel together.
# tools/check_platform_consistency.py fails if the two ever disagree.
PLATFORM_JSON_NAME = "platform.json"


def address_space_from_platform_json(path: "str | pathlib.Path") -> DeviceAddressSpace:
    """Build the address space from a bitstream's emitted platform.json."""
    facts = json.loads(pathlib.Path(path).read_text())
    missing = {"copy_base", "copy_size", "kernel_base", "kernel_size"} - facts.keys()
    if missing:
        raise ValueError(f"{path}: platform.json is missing {sorted(missing)}")
    return DeviceAddressSpace(
        copy_base=int(facts["copy_base"]),
        copy_size=int(facts["copy_size"]),
        kernel_base=int(facts["kernel_base"]),
        kernel_size=int(facts["kernel_size"]),
        kernel_alias_copy_base=(
            None if facts.get("kernel_alias_copy_base") is None
            else int(facts["kernel_alias_copy_base"])
        ),
    )


def load_packaged_address_space(
    search_dir: "str | pathlib.Path",
) -> DeviceAddressSpace | None:
    """Return the packaged map for a bitstream directory, or None if absent.

    Absent is not an error: packages built before platform.json existed still
    run on the built-in constants.
    """
    candidate = pathlib.Path(search_dir) / PLATFORM_JSON_NAME
    try:
        present = candidate.is_file()
    except OSError:
        # An unreadable or unmountable package directory must degrade to the
        # built-in constants, not abort a launch.
        return None
    return address_space_from_platform_json(candidate) if present else None


# Default to ZCU104 address space on ZCU104 packages.
_ADDRESS_SPACE: DeviceAddressSpace | None = ZCU104_ADDRESS_SPACE


def minitpuConfigureAddressSpace(address_space: DeviceAddressSpace) -> None:
    """Install the process-wide board map used by allocation and COPY."""
    if not isinstance(address_space, DeviceAddressSpace):
        raise TypeError("address_space must be a DeviceAddressSpace")
    global _ADDRESS_SPACE
    _ADDRESS_SPACE = address_space


def minitpuGetAddressSpace() -> DeviceAddressSpace:
    if _ADDRESS_SPACE is None:
        raise RuntimeError(
            "device address space is not configured; pass address_space= to "
            "open_device() or call minitpuConfigureAddressSpace()"
        )
    return _ADDRESS_SPACE


def _device_to_copy_address(addr: int, nbytes: int = 0) -> int:
    if _ADDRESS_SPACE is None:
        # Preserve raw-address operation for low-level diagnostics.
        return addr
    return _ADDRESS_SPACE.to_copy(addr, nbytes)
