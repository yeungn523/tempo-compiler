"""Provides kernel launch over the ABI LAUNCH command.

Kernel identity is the base and byte size of a binary already resident in device memory, so there is no registry,
handle, grid or slot index.
"""

from __future__ import annotations

from collections.abc import Sequence

from minitpu.abi import AXI_REG, LAUNCH_DESCRIPTOR, LAUNCH_ERROR_BIT, CmdOpc, build_cmd_words

from .backend import Mmio
from .memcpy import _backend, _ring_command

IRAM_DEPTH: int = 4096
"""Instruction-word depth of the on-chip IRAM, mirroring the localparam in src/pkg/minitpu_config_pkg.sv."""

INSTR_BYTES: int = 16
"""Byte size of one 128-bit VLIW bundle."""

_KBIN_ADDR_ALIGN: int = 32
"""Byte alignment the kernel base requires.

The pre-launch iram_loader latches ``device_mem_ptr = kbin_addr[31:5]`` and streams whole 256-bit beats, so a base
below this alignment has its low address bits silently dropped.
"""

_KBIN_ADDR_BITS: int = 32
"""Width of the device address the loader is wired to. The MINITPU wires only kbin_addr[31:0], so a base or end
past this width wraps silently."""

_IRAM_DEPTH_BYTES: int = IRAM_DEPTH * INSTR_BYTES
"""Byte capacity of the IRAM, past which the loader clamps the load and raises load_error."""

_KERNEL_ARG_CSR_OFFSETS: tuple[int, ...] = tuple(AXI_REG[f"kernel_arg_{slot}"] for slot in range(4))
"""AXI-Lite offsets of the kernel argument CSR slots the host writes before a launch. Each is a write alias whose
read side is an existing status word, so the read map is unchanged."""

DMA_ARG_SLOT_COUNT: int = len(_KERNEL_ARG_CSR_OFFSETS)
"""Number of kernel argument slots the on-chip CSR holds."""


def minitpuLaunchKernel(
    kbin_addr: int,
    kbin_size: int,
    *,
    args: Sequence[int] | None = None,
) -> None:
    """Issues a LAUNCH command for a kernel binary resident in device memory.

    Argument values are written to the on-chip CSR before the doorbell, so the slots are resident when the first
    argument-mode DMA operation issues. The sticky launch error bit is read afterwards, because the command
    processor retires an aborted launch exactly like a successful one.

    Args:
        kbin_addr: The device-memory base of the kernel binary, aligned to the loader's beat width.
        kbin_size: The byte size of the kernel binary, being a positive whole number of VLIW bundles.
        args: Up to four unsigned 32-bit device pointers or scalar strides to bind, or None for a baked binary.

    Raises:
        ValueError: When the size is not a positive multiple of the bundle size, the binary exceeds the IRAM
            capacity, the address range falls outside the width the loader is wired to or the alignment it
            requires, or more argument values are given than the CSR holds.
        RuntimeError: When the device aborts the launch, which the sticky launch error bit reports.
    """
    # Rejects a malformed launch up front, so it never rings the doorbell.
    if kbin_size <= 0:
        message = f"Unable to launch a kernel. The binary size must be positive, but got {kbin_size}."
        raise ValueError(message)
    if kbin_size % INSTR_BYTES:
        message = (
            f"Unable to launch a kernel. The binary size must be a multiple of {INSTR_BYTES} bytes, being one "
            f"128-bit VLIW bundle, but got {kbin_size}."
        )
        raise ValueError(message)
    if kbin_size > _IRAM_DEPTH_BYTES:
        message = (
            f"Unable to launch a kernel. The binary must fit the IRAM capacity of {IRAM_DEPTH} instruction words "
            f"({_IRAM_DEPTH_BYTES} bytes), but got {kbin_size} bytes, which the loader would clamp and flag as a "
            f"load error."
        )
        raise ValueError(message)
    if kbin_addr < 0:
        message = f"Unable to launch a kernel. The binary base must be non-negative, but got {kbin_addr}."
        raise ValueError(message)
    if (kbin_addr + kbin_size) > (1 << _KBIN_ADDR_BITS):
        message = (
            f"Unable to launch a kernel. The device address range must fit the {_KBIN_ADDR_BITS}-bit address the "
            f"loader is wired to, but got [0x{kbin_addr:x}, 0x{kbin_addr + kbin_size:x}). A base at or above "
            f"0x{1 << _KBIN_ADDR_BITS:x} loads from the truncated low bits, so stage the kernel in the low 4 GiB."
        )
        raise ValueError(message)
    address_offset = kbin_addr % _KBIN_ADDR_ALIGN
    if address_offset:
        message = (
            f"Unable to launch a kernel. The binary base must be {_KBIN_ADDR_ALIGN}-byte aligned, because the "
            f"pre-launch iram_loader maps kbin_addr[31:5] to a 256-bit word address, but got "
            f"0x{kbin_addr:x}, which is off by {address_offset}."
        )
        raise ValueError(message)

    mmio: Mmio = _backend()

    if args is not None:
        if len(args) > DMA_ARG_SLOT_COUNT:
            message = (
                f"Unable to launch a kernel. At most {DMA_ARG_SLOT_COUNT} values fit the argument CSR slots "
                f"0 to {DMA_ARG_SLOT_COUNT - 1}, but got {len(args)}."
            )
            raise ValueError(message)
        for slot, value in enumerate(args):
            argument = int(value)
            if not (0 <= argument <= 0xFFFF_FFFF):
                message = (
                    f"Unable to launch a kernel. Argument slot {slot} must hold an unsigned 32-bit value, since "
                    f"device pointers and strides are 32-bit in this DMEM aperture, but got 0x{argument:x}."
                )
                raise ValueError(message)
            mmio.write32(_KERNEL_ARG_CSR_OFFSETS[slot], argument)

    words = build_cmd_words(
        LAUNCH_DESCRIPTOR,
        {
            "opc": CmdOpc.LAUNCH,
            "kbin_addr_lo": kbin_addr & 0xFFFF_FFFF,
            "kbin_addr_hi": (kbin_addr >> 32) & 0xFFFF_FFFF,
            "kbin_size": kbin_size,
        },
    )
    _ring_command(mmio, words)

    status = mmio.read32(AXI_REG["launch_status"])
    if status & LAUNCH_ERROR_BIT:
        message = (
            f"Unable to complete the launch. The device aborted it with launch_status=0x{status:08X}, whose bit 0 "
            f"is the launch error. The iram_loader flagged a load error for kbin_addr=0x{kbin_addr:x} "
            f"kbin_size={kbin_size}, so the command processor never started the tile and device results are "
            f"stale. The binary is either oversize past {IRAM_DEPTH} instruction words or loaded no words."
        )
        raise RuntimeError(message)
