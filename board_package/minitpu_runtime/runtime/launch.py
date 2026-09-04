"""Kernel launch over the ABI LAUNCH command.

Public surface:
  - :func:`minitpuLaunchKernel`- issue a LAUNCH command for a kernel binary already
                                 resident in device memory.

Kernel identity is ``{kbin_addr, kbin_size}``: the host copies a 128-bit VLIW
binary to device memory, then LAUNCHes it by base + size.  There is no kernel
registry, handle, grid, or slot index in this runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

from minitpu.abi import AXI_REG, LAUNCH_DESCRIPTOR, LAUNCH_ERROR_BIT, CmdOpc, build_cmd_words

from .backend import Mmio
from .memcpy import _backend, _ring_command

# Mirrored from src/pkg/minitpu_config_pkg.sv (localparam int IRAM_DEPTH).
# hwparams.py was retired: it claimed to generate minitpu_config_pkg.sv via a
# build step that does not exist in this repository, and had drifted from the
# RTL on nearly every constant it declared. Of everything it exported the
# runtime used exactly this one name, so it is hardcoded here rather than
# kept alive as a stale mirror of an entire package. Unifying RTL constants
# with their software mirrors is tracked as CONST_MGMT work.
IRAM_DEPTH = 4096

# LAUNCH kernel-binary alignment (mirrors memcpy.CDMA_ALIGN).  The pre-launch
# iram_loader mini-DMA latches ``device_mem_ptr = kbin_addr[31:DEVICE_MEM_BYTE_SHIFT]``
# (iram_loader.sv:115-116; DEVICE_MEM_BYTE_SHIFT = clog2(DM_DATA_WIDTH/8) = 5 for
# the 256-bit device-memory word) and streams whole 256-bit beats, so the kernel
# base MUST be 32-byte aligned or the low address bits are silently dropped.
# The binary is a whole number of 128-bit VLIW bundles, so ``kbin_size`` must be
# a positive multiple of 16 bytes.  The runtime ENFORCES both up front
# (ValueError before the doorbell), exactly as memcpy guards CDMA_ALIGN, so a
# malformed LAUNCH never reaches the loader.
#
# Was 64, derived from the retired 512-bit DM word.  BF16-native narrowed the DM
# word to 256 bits, which both halves this requirement and makes it equal to
# memcpy.CDMA_ALIGN, so the two guards now agree by construction rather than by
# coincidence.
KBIN_ADDR_ALIGN = 32  # bytes, 256-bit device-memory read port
INSTR_BYTES = 16  # one 128-bit VLIW bundle

# The MINITPU wires only kbin_addr[31:0] to the loader (minitpu.sv:496; the high half
# minitpu.sv:524 is sunk into the lint concat), so the byte base AND its end must fit
# in 32 bits or the address silently wraps (0x1_0000_1000 would load from 0x1000).
KBIN_ADDR_BITS = 32
# The loader clamps to IRAM_DEPTH words and raises load_error (iram_loader.sv:70-77);
# reject an oversize kernel host-side with a clear cause instead of a clamped load.
IRAM_DEPTH_BYTES = IRAM_DEPTH * INSTR_BYTES  # 4096 words x 16 bytes = 64 KiB

# RFC P1 (kernarg arg-mode): the 4-entry kernel_arg CSR (param slots 0..3) the
# host writes before LAUNCH so an arg-mode kernel binds its DRAM base/stride from
# runtime values.  The offsets are named in abi.AXI_REG (kernel_arg_0..3 at
# 0x24/0x28/0x2C/0x30, minitpu_slave_axi_lite.v slv_reg9..12) -- WRITE aliases whose
# READ side is a pre-existing RO status word (0x24 READ stays dma_slot_done,
# 0x28/0x2C stay copy_status/copy_cdmasr; 0x30 is a fresh WO slot).  Derive the
# tuple from the SSOT rather than inlining raw hex (test_runtime_imports_from_abi
# forbids raw MINITPU offsets in runtime code).  Additive at/above 0x24 with an
# unchanged read map -> no ABI_VERSION bump.  Board-facing: unverified in the tile
# cocotb DV (which drives kernel_arg_csr directly).
KERNEL_ARG_CSR_OFFSETS = tuple(
    AXI_REG[f"kernel_arg_{slot}"] for slot in range(4)
)  # param slots 0..3 (u32 each)
DMA_ARG_SLOT_COUNT = len(KERNEL_ARG_CSR_OFFSETS)  # P1 4-entry CSR depth


def minitpuLaunchKernel(
    kbin_addr: int,
    kbin_size: int,
    *,
    args: Sequence[int] | None = None,
) -> None:
    """Issue a LAUNCH command for a kernel binary resident in device memory.

    RFC P1 (kernarg arg-mode): ``args`` is an optional sequence of up to
    :data:`DMA_ARG_SLOT_COUNT` (4) unsigned 32-bit param values (device pointers
    and/or scalar strides) bound to the on-chip param CSR (slots 0..3) BEFORE the
    LAUNCH doorbell.  An arg-mode DMA op then sources its DRAM base/stride from a
    param slot by index (the compiled binary carries slot indices, never
    addresses).  Each value is written to its :data:`KERNEL_ARG_CSR_OFFSETS`
    entry over MMIO.  ``args=None`` writes nothing (a baked/legacy kernel).
    NOTE (Stage-2): this arg-CSR write path is board-only; it is not exercised by
    the tile cocotb DV, which drives kernel_arg_csr as a direct input.

    ``kbin_addr`` / ``kbin_size`` are the device-memory base and byte size of the
    kernel binary (the only v1-active LAUNCH fields).  Builds the LAUNCH
    descriptor from ``abi.LAUNCH_DESCRIPTOR``, then runs the two-phase doorbell
    handshake (poll ``instr_ready`` for room, ring, poll for completion).

    Alignment guard (mirrors :func:`minitpuMemcpy`): ``kbin_addr`` must be
    :data:`KBIN_ADDR_ALIGN` (64-byte) aligned and ``kbin_size`` a positive
    multiple of :data:`INSTR_BYTES` (16), because the pre-launch iram_loader
    mini-DMA maps ``kbin_addr[31:6]`` to a 512-bit word address and streams whole
    128-bit VLIW bundles.  A violation raises :class:`ValueError` BEFORE the
    doorbell, so a malformed kernel never reaches the loader.

    Device-side abort: after the command retires, the sticky ``launch_error`` bit
    of the RO ``launch_status`` CSR is read and a :class:`RuntimeError` raised if
    set.  Retirement does not imply the kernel ran - the command_unit
    retires an aborted LAUNCH exactly like a successful one - so this read is the
    only thing separating "kernel ran" from "kernel silently never ran".
    """
    # Alignment / size guard (512-bit device-memory read port; see the module
    # constants).  Reject up front so a malformed LAUNCH never rings the doorbell.
    if kbin_size <= 0:
        raise ValueError(f"kbin_size must be > 0 (a non-empty kernel), got {kbin_size}")
    if kbin_size % INSTR_BYTES:
        raise ValueError(
            f"kbin_size must be a multiple of {INSTR_BYTES} bytes "
            f"(one 128-bit VLIW bundle), got {kbin_size}"
        )
    if kbin_size > IRAM_DEPTH_BYTES:
        raise ValueError(
            f"kbin_size {kbin_size} bytes exceeds the IRAM capacity of "
            f"{IRAM_DEPTH} instruction words ({IRAM_DEPTH_BYTES} bytes); the "
            f"loader would clamp and flag load_error. Shrink the kernel."
        )
    if kbin_addr < 0:
        raise ValueError(f"kbin_addr must be non-negative, got {kbin_addr}")
    # The loader only receives kbin_addr[31:0]; the base and its end must fit in
    # 32 bits or the read wraps silently.  kbin_size is already validated above.
    if (kbin_addr + kbin_size) > (1 << KBIN_ADDR_BITS):
        raise ValueError(
            f"LAUNCH device address range [0x{kbin_addr:x}, "
            f"0x{kbin_addr + kbin_size:x}) exceeds the {KBIN_ADDR_BITS}-bit "
            f"device-memory address the loader wires (minitpu.sv wires only "
            f"kbin_addr[31:0]); a base >= 0x{1 << KBIN_ADDR_BITS:x} loads from "
            f"the truncated low bits. Stage the kernel in the low 4 GiB."
        )
    addr_off = kbin_addr % KBIN_ADDR_ALIGN
    if addr_off:
        raise ValueError(
            f"LAUNCH requires a {KBIN_ADDR_ALIGN}-byte-aligned kbin_addr "
            f"(pre-launch iram_loader maps kbin_addr[31:6] to a 512-bit word "
            f"address): kbin_addr=0x{kbin_addr:x} (off {addr_off}). "
            f"Align to {KBIN_ADDR_ALIGN} bytes."
        )

    mmio: Mmio = _backend()

    # RFC P1: bind runtime param values into the 4-entry arg CSR before the
    # doorbell, so param_csr is resident when the first arg-mode DMA op issues.
    if args is not None:
        if len(args) > DMA_ARG_SLOT_COUNT:
            raise ValueError(
                f"minitpuLaunchKernel: at most {DMA_ARG_SLOT_COUNT} args fit the P1 "
                f"arg CSR (param slots 0..{DMA_ARG_SLOT_COUNT - 1}); got {len(args)}. "
                f"The DRAM param buffer + PARAM RF (>4 args) is P2."
            )
        for slot, value in enumerate(args):
            v = int(value)
            if not (0 <= v <= 0xFFFF_FFFF):
                raise ValueError(
                    f"arg slot {slot} value 0x{v:x} is not a u32 (device pointers "
                    f"and strides are 32-bit in the v1 DMEM aperture)"
                )
            mmio.write32(KERNEL_ARG_CSR_OFFSETS[slot], v)

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

    # Completion alone is NOT success.  When the loader flags load_error the CP
    # aborts the LAUNCH (never starts the tile) and retires normally, so
    # instr_ready returns to 1 and the _ring_command poll above passes on a
    # kernel that never ran.  The host guards upstream reject the cases we can
    # see, but the CMD window is raw MMIO and the RTL guard is the backstop, so
    # read the sticky bit the CP sets rather than reporting a silent success.
    status = mmio.read32(AXI_REG["launch_status"])
    if status & LAUNCH_ERROR_BIT:
        raise RuntimeError(
            f"LAUNCH aborted by the device: launch_status=0x{status:08X} "
            f"(bit 0 = launch_error). The iram_loader flagged load_error for "
            f"kbin_addr=0x{kbin_addr:x} kbin_size={kbin_size}, so the command "
            f"processor did not start the tile and the kernel never ran. The "
            f"kernel binary is oversize (> {IRAM_DEPTH} instruction words) or "
            f"loaded no words; device results are stale."
        )
