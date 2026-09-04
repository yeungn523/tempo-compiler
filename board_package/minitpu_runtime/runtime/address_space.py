"""Provides the board address map that translates kernel pointers into control-plane COPY addresses.

The authoritative map ships beside a bitstream as platform.json, and the constants here are the fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_PLATFORM_JSON_NAME: str = "platform.json"
"""Name of the address-map file the Vivado build emits beside a bitstream."""


@dataclass(frozen=True)
class DeviceAddressSpace:
    """Describes the two AXI views of one DMEM aperture.

    Runtime device pointers use the kernel view and COPY descriptors use the COPY view, so every device pointer
    crossing into a descriptor passes through ``to_copy``. AXI SmartConnect forwards a decoded address truncated to
    the slave port width rather than subtracting the segment base, so two apertures at different segment offsets
    address different DRAM bytes rather than aliasing each other.
    """

    copy_base: int
    """First COPY (CDMA) address of the aperture."""
    copy_size: int
    """Byte size of the COPY aperture, whose tail beyond the kernel view is raw COPY storage."""
    kernel_base: int
    """First address the MINITPU's kernel-data master sees."""
    kernel_size: int
    """Byte size of the kernel-visible aperture."""
    reserved: int = 0x0010_0000
    """Byte prefix of the kernel aperture held for instruction-image staging."""
    kernel_alias_copy_base: int | None = None
    """COPY address the kernel base refers to in DRAM, required where the two masters map the slave at different
    segment offsets and defaulting to ``copy_base`` where they do not."""

    def __post_init__(self) -> None:
        """Rejects an aperture pair that cannot describe a physical board.

        Raises:
            ValueError: When a base or size is negative, a size is not positive, the kernel aperture exceeds the
                COPY aperture it is backed by or the address width it is wired to, the reserved prefix falls
                outside the kernel aperture, or the kernel aperture's COPY alias falls outside the COPY aperture.
        """
        if min(self.copy_base, self.copy_size, self.kernel_base, self.kernel_size) < 0:
            message = "Unable to build the address space. Every aperture base and size must be non-negative."
            raise ValueError(message)
        if self.copy_size <= 0 or self.kernel_size <= 0:
            message = (
                f"Unable to build the address space. Both aperture sizes must be positive, but got "
                f"copy_size={self.copy_size} and kernel_size={self.kernel_size}."
            )
            raise ValueError(message)
        if self.kernel_size > self.copy_size:
            message = (
                f"Unable to build the address space. The kernel aperture must fit the backing COPY aperture, but "
                f"got kernel_size={self.kernel_size} against copy_size={self.copy_size}."
            )
            raise ValueError(message)
        if not (0 <= self.reserved < self.kernel_size):
            message = (
                f"Unable to build the address space. The reserved prefix must lie within the kernel aperture, but "
                f"got reserved={self.reserved} against kernel_size={self.kernel_size}."
            )
            raise ValueError(message)
        if self.copy_base + self.copy_size > (1 << 64):
            message = (
                f"Unable to build the address space. The COPY aperture must fit an unsigned 64-bit address, but it "
                f"ends at 0x{self.copy_base + self.copy_size:x}."
            )
            raise ValueError(message)
        if self.kernel_base + self.kernel_size > (1 << 32):
            message = (
                f"Unable to build the address space. The kernel aperture must fit the 32-bit MINITPU address "
                f"space, but it ends at 0x{self.kernel_base + self.kernel_size:x}."
            )
            raise ValueError(message)
        if not self._contains(self.copy_base, self.copy_size, self.kernel_alias_base, self.kernel_size):
            message = (
                f"Unable to build the address space. The kernel aperture's COPY alias "
                f"[0x{self.kernel_alias_base:x}, 0x{self.kernel_alias_base + self.kernel_size:x}) must lie inside "
                f"the COPY aperture [0x{self.copy_base:x}, 0x{self.copy_base + self.copy_size:x})."
            )
            raise ValueError(message)

    @property
    def kernel_alias_base(self) -> int:
        """Returns the COPY address of the first byte the MINITPU sees at the kernel base."""
        return self.copy_base if self.kernel_alias_copy_base is None else self.kernel_alias_copy_base

    def contains_kernel(self, address: int, nbytes: int = 0) -> bool:
        """Returns True when the given range lies inside the kernel-visible aperture."""
        return self._contains(self.kernel_base, self.kernel_size, address, nbytes)

    def contains_copy(self, address: int, nbytes: int = 0) -> bool:
        """Returns True when the given range lies inside the COPY aperture."""
        return self._contains(self.copy_base, self.copy_size, address, nbytes)

    def to_copy(self, address: int, nbytes: int = 0) -> int:
        """Translates a kernel pointer into the COPY address that reaches the same bytes.

        A range already expressed as a raw COPY address is returned unchanged.

        Raises:
            ValueError: When the range lies outside both apertures.
        """
        if self.contains_kernel(address, nbytes):
            return self.kernel_alias_base + address - self.kernel_base
        if self.contains_copy(address, nbytes):
            return address
        message = (
            f"Unable to translate a device range to a COPY address. The range [0x{address:x}, "
            f"0x{address + nbytes:x}) must lie inside the kernel aperture [0x{self.kernel_base:x}, "
            f"0x{self.kernel_base + self.kernel_size:x}) or the COPY aperture [0x{self.copy_base:x}, "
            f"0x{self.copy_base + self.copy_size:x})."
        )
        raise ValueError(message)

    def to_kernel(self, address: int, nbytes: int = 0) -> int:
        """Translates a COPY address into the kernel pointer that reaches the same bytes.

        Raises:
            ValueError: When the range has no kernel-visible alias.
        """
        if self.contains_kernel(address, nbytes):
            return address
        if self.contains_copy(address, nbytes):
            offset = address - self.kernel_alias_base
            if 0 <= offset and offset + nbytes <= self.kernel_size:
                return self.kernel_base + offset
        message = (
            f"Unable to translate a COPY range to a kernel pointer. The range [0x{address:x}, "
            f"0x{address + nbytes:x}) has no kernel-visible alias."
        )
        raise ValueError(message)

    @staticmethod
    def _contains(base: int, size: int, address: int, nbytes: int) -> bool:
        """Returns True when the range starting at ``address`` lies inside the aperture at ``base``."""
        return nbytes >= 0 and address >= base and address <= base + size and nbytes <= base + size - address


ZCU102_ADDRESS_SPACE = DeviceAddressSpace(
    copy_base=0xA000_0000,
    copy_size=0x2000_0000,
    kernel_base=0xA000_0000,
    kernel_size=0x2000_0000,
)
"""Fallback map for the ZCU102, whose two masters see the DMEM at the same offset."""

ZCU104_ADDRESS_SPACE = DeviceAddressSpace(
    copy_base=0x4_0000_0000,
    copy_size=0x4_0000_0000,
    kernel_base=0x8000_0000,
    kernel_size=0x8000_0000,
    kernel_alias_copy_base=0x4_8000_0000,
)
"""Fallback map for the ZCU104, whose two masters reach one DDR4 slave through segments at different offsets.

SmartConnect truncates a decoded address instead of subtracting the segment base, so the kernel base aliases COPY
0x4_8000_0000 and not the COPY base. A wrong alias fails silently: the COPY completes and the LAUNCH executes
uninitialised DRAM 2 GiB away.
"""

_ADDRESS_SPACE: DeviceAddressSpace | None = ZCU104_ADDRESS_SPACE
"""Process-wide map used by allocation and COPY, defaulting to the board these packages ship for."""


def address_space_from_platform_json(path: str | Path) -> DeviceAddressSpace:
    """Builds the address space from the platform.json a bitstream build emitted.

    Raises:
        ValueError: When the file omits any of the four required aperture facts.
    """
    facts = json.loads(Path(path).read_text())
    missing = {"copy_base", "copy_size", "kernel_base", "kernel_size"} - facts.keys()
    if missing:
        message = f"Unable to read the address space from {path}. The file is missing {sorted(missing)}."
        raise ValueError(message)
    return DeviceAddressSpace(
        copy_base=int(facts["copy_base"]),
        copy_size=int(facts["copy_size"]),
        kernel_base=int(facts["kernel_base"]),
        kernel_size=int(facts["kernel_size"]),
        kernel_alias_copy_base=(
            None if facts.get("kernel_alias_copy_base") is None else int(facts["kernel_alias_copy_base"])
        ),
    )


def load_packaged_address_space(search_dir: str | Path) -> DeviceAddressSpace | None:
    """Loads the map packaged with a bitstream directory.

    Returns:
        The packaged address space, or None when the directory carries no platform.json, since a package built
        before that file existed runs on the built-in constants.
    """
    candidate = Path(search_dir) / _PLATFORM_JSON_NAME
    try:
        present = candidate.is_file()
    except OSError:
        # Degrades to the built-in constants rather than aborting a launch on an unreadable package directory.
        return None
    return address_space_from_platform_json(candidate) if present else None


def minitpuConfigureAddressSpace(address_space: DeviceAddressSpace) -> None:
    """Installs the process-wide board map used by allocation and COPY.

    Raises:
        TypeError: When the argument is not a DeviceAddressSpace.
    """
    if not isinstance(address_space, DeviceAddressSpace):
        message = (
            f"Unable to configure the address space. The map must be a DeviceAddressSpace, but got "
            f"{type(address_space).__name__}."
        )
        raise TypeError(message)
    global _ADDRESS_SPACE
    _ADDRESS_SPACE = address_space


def minitpuGetAddressSpace() -> DeviceAddressSpace:
    """Returns the process-wide board map.

    Raises:
        RuntimeError: When no map is configured.
    """
    if _ADDRESS_SPACE is None:
        message = (
            "Unable to return the device address space. No map is configured, so pass address_space= to "
            "open_device() or call minitpuConfigureAddressSpace()."
        )
        raise RuntimeError(message)
    return _ADDRESS_SPACE


def _device_to_copy_address(address: int, nbytes: int = 0) -> int:
    """Translates a device pointer for a COPY descriptor, passing it through unchanged when no map is configured."""
    if _ADDRESS_SPACE is None:
        # Passes the address through, which keeps raw-address operation available for low-level diagnostics.
        return address
    return _ADDRESS_SPACE.to_copy(address, nbytes)
