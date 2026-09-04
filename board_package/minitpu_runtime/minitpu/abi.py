"""Defines the host to device ABI: the command window, the command opcodes, the register map and the descriptor and
patch-table wire formats.

The bitstream mirrors these constants through the generated SystemVerilog packages, so changing a layout, an opcode, a
unit or a field requires an ABI_VERSION bump and a rebuild.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import NamedTuple

ABI_VERSION: int = 9
"""Interface version baked into the bitstream.

No register reports it back and the runtime checks nothing, so it marks a required rebuild by convention rather than
catching a stale bitstream. Read-only registers added above 0x24 keep the current version.
"""

BUILD_ID_MAGIC: int = 0xB1
"""Fixed tag in bits [31:24] of BUILD_ID. A bitstream without the build_id register reads 0 at that offset, so the tag
check separates a real device identity from an absent one.
"""

BUILD_ID: int = BUILD_ID_MAGIC << 24
"""Device identity read from the build_id register, holding BUILD_ID_MAGIC in [31:24] and the source commit SHA-1
prefix in [23:0].

The tracked value is the bare tag. A board build stamps its own copy from a clean HEAD and records the full SHA-1 in
build_identity/build_identity.json, which resolves the truncated hardware value.
"""

AXI_REG: dict[str, int] = {
    "cmd_w0": 0x00,  # WO: header, [3:0] opcode, [4] doorbell, [7:5] direction/target, [31:8] reserved
    "cmd_w1": 0x04,  # WO: COPY host_phys[31:0] / LAUNCH kbin_addr[31:0]
    "cmd_w2": 0x08,  # WO: COPY host_phys[63:32] / LAUNCH kbin_addr[63:32]
    "cmd_w3": 0x0C,  # WO: COPY dev_addr[31:0] / LAUNCH kbin_size[31:0] in bytes
    "cmd_w4": 0x10,  # WO: COPY dev_addr[63:32] / LAUNCH kernarg_ptr[31:0], reserved
    "cmd_w5": 0x14,  # WO: COPY len[31:0] in bytes / LAUNCH kernarg_ptr[63:32], reserved
    "cmd_w6": 0x18,  # WO: COPY reserved / LAUNCH grid_x[31:0], reserved
    "cmd_w7": 0x1C,  # WO: COPY reserved / LAUNCH grid_y[31:0], reserved
    "instr_ready": 0x20,  # RO: 1 when every sub-FSM is idle
    "dma_slot_done": 0x24,  # RO: async-DMA slot-done bits [1:0]
    "kernel_arg_0": 0x24,  # WO: parameter slot 0, write alias over the dma_slot_done read
    "kernel_arg_1": 0x28,  # WO: parameter slot 1, write alias over the copy_status read
    "kernel_arg_2": 0x2C,  # WO: parameter slot 2, write alias over the copy_cdmasr read
    "kernel_arg_3": 0x30,  # WO: parameter slot 3, with no read-side alias
    "copy_status": 0x28,  # RO: [0] err, [1] oob, [2] timeout, [3] done, [4] busy, [5] reset_timeout, [15:8] transfers
    "copy_cdmasr": 0x2C,  # RO: last CDMASR value the copy engine read while polling
    "launch_status": 0x3C,  # RO: [0] sticky launch_error
    "perf_cnt_cycles": 0x40,  # RO: free-running core cycle counter
    "perf_cnt_instrs": 0x44,  # RO: count of bundles issued
    "dma_err": 0x4C,  # RO: sticky core DMA fault, cause and observation bits
    "beat_count": 0x50,  # RO: free-running count of 256-bit device-memory words moved
    "dma_overlap": 0x54,  # RO: [7:0] peak concurrent non-idle slots, [31:8] cycles with two or more non-idle
    "build_id": 0x58,  # RO: baked device identity, laid out by BUILD_ID
}
"""Maps each ABI register name to its AXI-Lite byte offset.

The command words at 0x00 through 0x1C are write-only and the words from 0x20 up are read-only status, except the four
kernel_arg slots, which are write aliases sharing addresses with status words.
"""

CMD_WINDOW_WORDS: int = 8

DOORBELL_BIT: int = 0x10
"""Self-clearing trigger in bit 4 of cmd_w0, cleared one to two cycles after the command unit latches the window."""

LAUNCH_ERROR_BIT: int = 0x1
"""Sticky flag in bit 0 of launch_status, set when the loader rejected the last kernel binary and cleared when the next
LAUNCH is accepted.

An aborted LAUNCH still retires with instr_ready back at 1, so a completion poll reports success and only this bit
tells the two apart. A copy-only bitstream carries no loader and leaves the register at 0, which the device identity
has to catch instead.
"""

DMA_SLOT_DONE_CSR: int = AXI_REG["dma_slot_done"]

PATCH_SECTION_MAGIC: bytes = b"MINITPUP"
"""Opens a patch section, followed by a u32 record count and that many 16-byte records."""

PATCH_HEADER_MAGIC: bytes = b"MINITPUH"
"""Opens a v2 binary, followed by a u32 byte offset to the patch section. A v1 binary carries no header, so its section
is found by scanning for PATCH_SECTION_MAGIC.
"""

PATCH_HEADER_SIZE: int = 8

EXPR_GRID_LINEAR: int = 0x01
"""The only patch expression kind: a grid axis scaled and offset by compile-time constants."""

GRID_AXIS_X: int = 0
GRID_AXIS_Y: int = 1
GRID_AXIS_Z: int = 2

_PATCH_RECORD_FMT: str = "<IBBBBIi"
"""Record layout: instruction index, operand slot, expression kind, grid axis, padding, scale, addend."""

PATCH_RECORD_SIZE: int = struct.calcsize(_PATCH_RECORD_FMT)

assert PATCH_RECORD_SIZE == 16, PATCH_RECORD_SIZE

_OPERAND_SLOT_SHIFTS: tuple[int, int, int] = (32, 16, 0)
"""Bit shift of op0, op1 and op2 within the 64-bit instruction word."""

# The v7 host-dispatch sub-FSM still decodes these mode values, so the constant generator keeps emitting the matching
# localparams to keep the modes package elaborating. No opcode maps to them and the runtime never writes them.
_LEGACY_RTL_MODES: dict[str, int] = {
    "IDLE": 0,
    "WRITE_BRAM": 1,
    "READ_BRAM": 2,
    "COMPUTE": 3,
    "WRITE_IRAM": 4,
}


class CmdOpc(IntEnum):
    """Defines the opcode in bits [3:0] of the header word, which selects the engine the command unit routes to."""

    COPY = 0x0
    LAUNCH = 0x1


class CmdDir(IntEnum):
    """Defines the COPY transfer direction, carried in bits [7:5] of the header word."""

    H2D = 0
    D2H = 1


class RegSpec(NamedTuple):
    """Defines one ABI register: its bit-width, its canonical units, its RW, RO or WO access mode, and a
    cross-reference to the RTL that implements it.
    """

    width: int
    units: str
    access: str = "RW"
    note: str = ""


class CmdField(NamedTuple):
    """Defines the placement of one named field in the command window."""

    word: int
    lo: int
    width: int
    name: str
    status: str = "v1"
    """Milestone that activates the field. The serving and v3 fields hold their place in the wire layout and the
    current RTL ignores them.
    """


class PatchRecord(NamedTuple):
    """Defines one grid-dependent instruction operand the host resolves before each launch.

    The compiler emits such operands as placeholders and appends a patch table, so one binary serves every grid without
    per-launch recompilation. A program carrying no records is launch-invariant.
    """

    insn_index: int
    operand_slot: int
    """Selects op0, op1 or op2 as 0, 1 or 2."""
    expr_kind: int = EXPR_GRID_LINEAR
    grid_axis: int = GRID_AXIS_X
    scale: int = 1
    addend: int = 0

    def pack(self) -> bytes:
        """Serializes the record into its 16-byte wire form, raising ValueError on a field that overflows its width."""
        if not (0 <= self.insn_index <= 0xFFFF_FFFF):
            message = (
                f"Unable to pack the patch record. The instruction index must fit an unsigned 32-bit integer, but "
                f"got {self.insn_index}."
            )
            raise ValueError(message)
        if self.operand_slot not in (0, 1, 2):
            message = (
                f"Unable to pack the patch record. The operand slot must be 0, 1 or 2, but got {self.operand_slot}."
            )
            raise ValueError(message)
        if self.expr_kind != EXPR_GRID_LINEAR:
            message = (
                f"Unable to pack the patch record. The expression kind must be EXPR_GRID_LINEAR (0x01), but got "
                f"{self.expr_kind:#x}."
            )
            raise ValueError(message)
        if self.grid_axis not in (GRID_AXIS_X, GRID_AXIS_Y, GRID_AXIS_Z):
            message = f"Unable to pack the patch record. The grid axis must be 0, 1 or 2, but got {self.grid_axis}."
            raise ValueError(message)
        if not (0 <= self.scale <= 0xFFFF_FFFF):
            message = (
                f"Unable to pack the patch record. The scale must fit an unsigned 32-bit integer, but got {self.scale}."
            )
            raise ValueError(message)
        if not (-(1 << 31) <= self.addend < (1 << 31)):
            message = (
                f"Unable to pack the patch record. The addend must fit a signed 32-bit integer, but got {self.addend}."
            )
            raise ValueError(message)
        return struct.pack(
            _PATCH_RECORD_FMT,
            self.insn_index,
            self.operand_slot,
            self.expr_kind,
            self.grid_axis,
            0,  # padding
            self.scale,
            self.addend,
        )

    @classmethod
    def unpack(cls, buffer: bytes) -> PatchRecord:
        """Deserializes one record from the leading PATCH_RECORD_SIZE bytes of the buffer."""
        instruction_index, operand_slot, expression_kind, grid_axis, _padding, scale, addend = struct.unpack(
            _PATCH_RECORD_FMT, buffer[:PATCH_RECORD_SIZE]
        )
        return cls(
            insn_index=instruction_index,
            operand_slot=operand_slot,
            expr_kind=expression_kind,
            grid_axis=grid_axis,
            scale=scale,
            addend=addend,
        )

    def resolve(self, grid: tuple[int, int, int]) -> int:
        """Evaluates the record against one launch grid, raising ValueError when the result overflows the 16-bit
        operand field it has to be written into.
        """
        value = grid[self.grid_axis] * self.scale + self.addend
        if not (0 <= value < (1 << 16)):
            message = (
                f"Unable to resolve the patch record. The value must fit the 16-bit operand field, but got {value} "
                f"for grid {grid} with axis {self.grid_axis}, scale {self.scale} and addend {self.addend}."
            )
            raise ValueError(message)
        return value


REG_SEMANTICS: dict[str, RegSpec] = {
    "cmd_w0": RegSpec(
        width=32,
        units="cmd_header",
        access="WO",
        note="[3:0] opcode, [4] doorbell, [7:5] direction/target, [31:8] reserved. Latched on the doorbell edge.",
    ),
    "cmd_w1": RegSpec(width=32, units="cmd_body", access="WO", note="COPY host_phys[31:0] / LAUNCH kbin_addr[31:0]."),
    "cmd_w2": RegSpec(width=32, units="cmd_body", access="WO", note="COPY host_phys[63:32] / LAUNCH kbin_addr[63:32]."),
    "cmd_w3": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="COPY dev_addr[31:0], an absolute device address / LAUNCH kbin_size[31:0] in bytes.",
    ),
    "cmd_w4": RegSpec(
        width=32, units="cmd_body", access="WO", note="COPY dev_addr[63:32] / LAUNCH kernarg_ptr[31:0], reserved."
    ),
    "cmd_w5": RegSpec(
        width=32, units="cmd_body", access="WO", note="COPY len[31:0] in bytes / LAUNCH kernarg_ptr[63:32], reserved."
    ),
    "cmd_w6": RegSpec(width=32, units="cmd_body", access="WO", note="COPY reserved / LAUNCH grid_x[31:0], reserved."),
    "cmd_w7": RegSpec(width=32, units="cmd_body", access="WO", note="COPY reserved / LAUNCH grid_y[31:0], reserved."),
    "instr_ready": RegSpec(
        width=1, units="status", access="RO", note="1 when every sub-FSM is idle, both completion and backpressure."
    ),
    "dma_slot_done": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="Bit i is set when the async DMA for slot i completes, and cleared when flush.slot exits.",
    ),
    "kernel_arg_0": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="Parameter slot 0, latched into kernel_arg_csr[31:0]. Write alias, the 0x24 read stays dma_slot_done.",
    ),
    "kernel_arg_1": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="Parameter slot 1, latched into kernel_arg_csr[63:32]. Write alias, the 0x28 read stays copy_status.",
    ),
    "kernel_arg_2": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="Parameter slot 2, latched into kernel_arg_csr[95:64]. Write alias, the 0x2C read stays copy_cdmasr.",
    ),
    "kernel_arg_3": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="Parameter slot 3, latched into kernel_arg_csr[127:96]. Write-only, with no read-side alias.",
    ),
    "copy_status": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note=(
            "Latched per COPY: [0] err, [1] oob, [2] timeout, [3] done, [4] busy, [5] reset_timeout, [15:8] "
            "transfers. An address off a 32-byte boundary raises DMAIntErr here and moves nothing."
        ),
    ),
    "launch_status": RegSpec(
        width=32, units="bitmask", access="RO", note="[0] sticky launch_error, described by LAUNCH_ERROR_BIT."
    ),
    "copy_cdmasr": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="Last CDMASR (PG034) read while polling: bit 1 Idle, bits 4/5/6 DMAIntErr/SlvErr/DecErr, bit 14 Err_Irq.",
    ),
    "dma_err": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note=(
            "Sticky from the first fault: [0] dma_err, [1] cause resp_error, [2] cause watchdog, [3] rsp_seen, [4] "
            "bridge_to, [5] ar_seen, [6] r_seen, [7] aw_seen, [15:8] err_slot, [17:16] err_resp, [18] b_seen. The "
            "bridge turns a silent slave into a synthetic SLVERR, so [2] is unreachable and [1] with [4] means the "
            "SLVERR was manufactured."
        ),
    ),
    "perf_cnt_cycles": RegSpec(
        width=32,
        units="cycles",
        access="RO",
        note="Free-running core cycle counter, reset on rst_n. The device-side time base, free of host overhead.",
    ),
    "perf_cnt_instrs": RegSpec(
        width=32,
        units="count",
        access="RO",
        note="Bundles issued, reset on rst_n. The ratio to perf_cnt_cycles is the achieved issue rate.",
    ),
    "beat_count": RegSpec(
        width=32,
        units="count",
        access="RO",
        note=(
            "Count of 256-bit device-memory words that left the DMA, reset on rst_n. Bandwidth is beat_count * 32 B / "
            "(perf_cnt_cycles / f_clk)."
        ),
    ),
    "dma_overlap": RegSpec(
        width=32,
        units="count",
        access="RO",
        note=(
            "[7:0] peak slots non-idle in one cycle, [31:8] cycles with two or more. A slot parked in S_DONE is "
            "excluded, so a serialised kernel cannot read as overlapped."
        ),
    ),
    "build_id": RegSpec(
        width=32,
        units="identity",
        access="RO",
        note="[31:24] magic 0xB1, [23:0] Git SHA-1 prefix. A build without this register reads 0 and fails the check.",
    ),
}

CMD_HEADER: tuple[CmdField, ...] = (
    CmdField(0, 0, 4, "opc"),
    CmdField(0, 4, 1, "doorbell"),
    CmdField(0, 5, 3, "dir_tgt"),
)
"""Header sub-fields of word 0, shared by both command kinds."""

# The 256-bit CDMA has no data realignment engine, so host_phys and dev_addr must be 32-byte aligned or the transfer
# raises DMAIntErr and moves nothing until a copy-engine soft reset. The runtime enforces that before issuing the
# descriptor. The length needs no alignment, because the CDMA handles a partial final beat.
COPY_DESCRIPTOR: tuple[CmdField, ...] = CMD_HEADER + (
    CmdField(1, 0, 32, "host_phys_lo"),
    CmdField(2, 0, 32, "host_phys_hi"),
    CmdField(3, 0, 32, "dev_addr_lo"),
    CmdField(4, 0, 32, "dev_addr_hi"),
    CmdField(5, 0, 32, "len"),
)
"""Wire layout of a COPY command, which requests a host to device memcpy."""

LAUNCH_DESCRIPTOR: tuple[CmdField, ...] = CMD_HEADER + (
    CmdField(1, 0, 32, "kbin_addr_lo"),
    CmdField(2, 0, 32, "kbin_addr_hi"),
    CmdField(3, 0, 32, "kbin_size"),
    CmdField(4, 0, 32, "kernarg_ptr_lo", "serving"),
    CmdField(5, 0, 32, "kernarg_ptr_hi", "serving"),
    CmdField(6, 0, 32, "grid_x", "v3"),
    CmdField(7, 0, 32, "grid_y", "v3"),
)
"""Wire layout of a LAUNCH command, which runs a kernel binary resident in device memory."""

CMD_DESCRIPTORS: dict[int, tuple[CmdField, ...]] = {
    CmdOpc.COPY: COPY_DESCRIPTOR,
    CmdOpc.LAUNCH: LAUNCH_DESCRIPTOR,
}


def build_cmd_words(descriptor: tuple[CmdField, ...], values: dict[str, int]) -> list[int]:
    """Packs named field values into the command window words, masking each to its declared width.

    The doorbell bit is left clear so the caller can stage W1 through W7 first and ring the doorbell as the final write
    to W0, which is what makes the command unit latch a complete window.

    Raises:
        ValueError: If a key names no field of the descriptor, or a value overflows its field width.
    """
    fields = {field.name: field for field in descriptor}
    unknown = set(values) - set(fields)
    if unknown:
        message = (
            f"Unable to build the command words. Every value key must name a field of the descriptor, but got the "
            f"unknown key(s) {sorted(unknown)} against the known fields {sorted(fields)}."
        )
        raise ValueError(message)
    words = [0] * CMD_WINDOW_WORDS
    for name, value in values.items():
        field = fields[name]
        if not (0 <= value < (1 << field.width)):
            message = (
                f"Unable to build the command words. The value for field '{name}' must fit {field.width} bits, but "
                f"got {value}."
            )
            raise ValueError(message)
        words[field.word] |= (value & ((1 << field.width) - 1)) << field.lo
    return words


def parse_binary(blob: bytes) -> tuple[list[int], list[PatchRecord]]:
    """Splits a compiler binary into its program words and patch records.

    A v2 blob opens with PATCH_HEADER_MAGIC and a byte offset to the patch section. A v1 blob carries no header, so the
    section is found by scanning and the program is treated as launch-invariant when the scan finds nothing.
    """
    if blob[:4] == PATCH_HEADER_MAGIC:
        (patch_offset,) = struct.unpack("<I", blob[4:8])
        instruction_bytes = blob[PATCH_HEADER_SIZE:patch_offset]
        patch_section = blob[patch_offset:]
    elif blob[:4] == PATCH_SECTION_MAGIC:
        message = "Unable to parse the binary. The input must be a whole binary blob, but got a bare patch section."
        raise ValueError(message)
    else:
        index = blob.find(PATCH_SECTION_MAGIC)
        if index < 0:
            instruction_bytes = blob
            patch_section = PATCH_SECTION_MAGIC + struct.pack("<I", 0)
        else:
            instruction_bytes = blob[:index]
            patch_section = blob[index:]

    word_count = len(instruction_bytes) // 8
    program_words = list(struct.unpack(f"<{word_count}Q", instruction_bytes[: word_count * 8]))
    return program_words, parse_patch_table(patch_section)


def parse_patch_table(section: bytes) -> list[PatchRecord]:
    """Parses a patch section into its records, rejecting any expression kind other than EXPR_GRID_LINEAR.

    Returns an empty list for a launch-invariant program.
    """
    if section[:4] != PATCH_SECTION_MAGIC:
        message = (
            f"Unable to parse the patch table. The section must open with {PATCH_SECTION_MAGIC!r}, but got "
            f"{section[:4]!r}."
        )
        raise ValueError(message)
    (count,) = struct.unpack("<I", section[4:8])
    records: list[PatchRecord] = []
    offset = 8
    for _ in range(count):
        record = PatchRecord.unpack(section[offset : offset + PATCH_RECORD_SIZE])
        if record.expr_kind != EXPR_GRID_LINEAR:
            message = (
                f"Unable to parse the patch table. The record at offset {offset} must declare EXPR_GRID_LINEAR, but "
                f"got expression kind {record.expr_kind:#x}. Variable loop bounds bind through inline kernel "
                f"arguments instead."
            )
            raise ValueError(message)
        records.append(record)
        offset += PATCH_RECORD_SIZE
    return records


def apply_patch_table(program_words: list[int], records: list[PatchRecord], grid: tuple[int, int, int]) -> list[int]:
    """Resolves every record for one launch grid and rewrites the affected operands in place.

    The 64-bit instruction word packs an 8-bit opcode, 8 flag bits and three 16-bit operands, so op0 occupies bits
    [47:32], op1 bits [31:16] and op2 bits [15:0].
    """
    for record in records:
        value = record.resolve(grid)
        shift = _OPERAND_SLOT_SHIFTS[record.operand_slot]
        word = program_words[record.insn_index]
        word &= ~(0xFFFF << shift) & 0xFFFF_FFFF_FFFF_FFFF
        word |= (value & 0xFFFF) << shift
        program_words[record.insn_index] = word
    return program_words
