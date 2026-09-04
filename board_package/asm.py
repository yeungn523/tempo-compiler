"""Provides the packaging interface for the MINITPU's physical 128-bit VLIW bundles, and the software scheduling
that makes a stream of them legal.

The machine carries no hazard interlocks. It issues one bundle per cycle and the delay field is the only fetch-head
hold, so every emitted image is spaced here before it reaches a board. The scheduling mirrors tb/bundle_scheduler.svh
and both take their latencies from docs/isa_latency.json.
"""

from __future__ import annotations

import functools
import hashlib
import os

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isa_model import (
    MATRIX_OCCUPANCY,
    MATRIX_PORT_A,
    MATRIX_RESULT_LATENCY,
    OPERATIONS,
    RTL_PARAMS,
)

_OPCODES: dict[str, int] = {
    mnemonic: opcode
    for group in OPERATIONS
    for mnemonic, opcode in zip(group["ops"], group["opcodes"])
}
"""Numeric opcode of every mnemonic the ISA contract names."""

_WRITEBACK: dict[str, int] = {
    mnemonic: group["w"]
    for group in OPERATIONS
    for mnemonic in group["ops"]
    if isinstance(group["w"], int)
}
"""Writeback offset of every mnemonic claiming a single cycle of the VREG write port."""

BUNDLE_BYTES: int = 16
"""Size of one physical VLIW bundle in bytes."""

BUNDLE_BITS: int = 128
"""Width of one physical VLIW bundle in bits."""

HALT_BUNDLE: int = 0x0000000200000000
"""Canonical F:halt word the current RTL encoder emits, carried as an exact physical word rather than re-encoded."""

# Mirrors sequencer_pkg::compute_payload_t ({is_dma, e, m, x, s, f, delay, reserved}, MSB first) at NUM_LANES=16
# and NUM_SUBLANES=4:
#
#   is_dma [127]   e [126:108]   m [107:101]   x [100:61]
#   s [60:34]      f [33:7]      delay [6:1]   reserved [0]
#
# Derives every position below from these widths, so a width change in sequencer_pkg moves every position that
# follows it.

E_SLOT_W: int = 19
"""Width of the E slot in bits."""

M_SLOT_W: int = 7
"""Width of the M slot in bits."""

X_SLOT_W: int = 40
"""Width of the X slot in bits."""

S_SLOT_W: int = 27
"""Width of the S slot in bits."""

F_SLOT_W: int = 27
"""Width of the F slot in bits."""

LEVEL_SEL_W: int = 2
"""Width of the AGU level selector in bits, covering the four nesting levels sequencer_pkg's STACK_DEPTH holds."""

DELAY_W: int = 6
"""Width of the delay field in bits."""

COMPUTE_RESERVED_W: int = (BUNDLE_BITS - 1) - (
    E_SLOT_W + M_SLOT_W + X_SLOT_W + S_SLOT_W + F_SLOT_W + DELAY_W)
"""Bits the compute payload leaves unassigned once every slot and the delay field are placed."""

NUM_SUBLANES: int = 4
"""Banks per lane, mirroring vpu_pkg's NUM_SUBLANES."""

SUBLANE_STRIDE_W: int = 7
"""Width of the X-slot stride immediate in bits."""

# Overlays one DMA descriptor on the storage the compute format spends on its E/M/X/S/F fields, with bit 127
# selecting between the two. A co-issued slot therefore reaches no execution unit: it rewrites the descriptor bits
# it overlaps, or it lands in the DMA format's reserved field and vanishes. Hardware does both silently, so
# assembly time is where they are caught.
DMA_EXCLUSIVE_SLOTS: frozenset[str] = frozenset({"e", "v", "m", "x", "s", "f"})
"""Compute slots a DMA bundle is unable to carry alongside its descriptor."""

MAX_DELAY: int = (1 << DELAY_W) - 1
"""Largest delay one bundle's field encodes, past which the scheduler inserts filler bundles."""

W_ALU: int = RTL_PARAMS["WB_W_ALU"]
"""Writeback offset of an ALU result, so a consumer is legal one cycle later."""

W_SFU: int = RTL_PARAMS["WB_W_SFU"]
"""Writeback offset of an SFU table result."""

W_REDUCE: int = RTL_PARAMS["WB_W_REDUCE"]
"""Writeback offset of a reduction tree result."""

W_VLD: int = RTL_PARAMS["WB_W_VLD"]
"""Writeback offset of a conflict-free VMEM load, which a split load extends by its conflict degree."""

W_TRANSPOSE: int = _WRITEBACK["vtxout"]
"""Writeback offset of a vtxout, covering one registered read stage inside the unit and the shared writeback
boundary every other source pays."""

W_MPOP_FIRST: int = RTL_PARAMS["WB_W_MPOP_FIRST"]
"""First cycle a vmatpop's writeback occupies, since a pop holds the port across its beats."""

W_MPOP_LAST: int = RTL_PARAMS["WB_W_MPOP_LAST"]
"""Last cycle a vmatpop's writeback occupies."""

W_SCALAR: int = 1
"""Scalar writeback reservation offset, so a consumer is legal at W_SCALAR + 1.

`sequencer_pkg` leaves SREG RAW to the compiler with nothing interlocking it, and every builder discharges it by
leaving two bundles between a scalar write and its use."""

LB_CAP: int = 24
"""Loop-buffer capacity in bundles, mirroring sequencer_loop_ctrl's LB_CAP. A longer body refetches from IRAM."""

DMA_SLOT_COUNT: int = 2
"""Descriptor slots a flush mask selects between, mirroring minitpu_config_pkg's DMA_SLOT_COUNT."""

_E_SHIFT: int = (BUNDLE_BITS - 1) - E_SLOT_W
"""Bit position the E slot starts at."""

_M_SHIFT: int = _E_SHIFT - M_SLOT_W
"""Bit position the M slot starts at."""

_X_SHIFT: int = _M_SHIFT - X_SLOT_W
"""Bit position the X slot starts at."""

_S_SHIFT: int = _X_SHIFT - S_SLOT_W
"""Bit position the S slot starts at."""

_F_SHIFT: int = _S_SHIFT - F_SLOT_W
"""Bit position the F slot starts at."""

_DELAY_SHIFT: int = _F_SHIFT - DELAY_W
"""Bit position the compute bundle's delay field starts at."""

_DMA_DELAY_SHIFT: int = 85
"""Bit position the DMA bundle's delay field starts at, which the descriptor layout places apart from the
compute one."""

assert _DELAY_SHIFT == COMPUTE_RESERVED_W, (
    "compute payload field positions disagree with the slot widths")

def _e_opcodes(*, reads_b: bool | None = None, w: int | None = None) -> set[int]:
    """Returns the E opcodes matching a second-operand read, a writeback offset, or both."""
    return {
        opcode
        for group in OPERATIONS
        for opcode in (group["opcodes"] if group["slot"] == "E" else ())
        if (reads_b is None or ("rb" in group["reads"]) == reads_b)
        and (w is None or group["w"] == w)
    }


_E_BINARY_OPS: set[int] = _e_opcodes(reads_b=True)
"""E opcodes reading both source registers."""

_E_ALU_OPS: set[int] = _e_opcodes(w=W_ALU)
"""E opcodes carrying ALU latency, which adds the unary vmov to the binary set.

vmax and vmin ride the ALU's move stages, so every opcode in the set keeps one latency."""

_E_TXOUT: int = _OPCODES["vtxout"]
"""E opcode draining the transpose tile, which writes a VREG and reads none."""

_F_FLUSH: int = _OPCODES["flush_slot"]
"""F opcode blocking until every DMA slot its mask names reports done."""

_E_SFU_OPS: set[int] = _e_opcodes(w=W_SFU)
"""E opcodes carrying SFU latency, each reading port A alone."""

_E_REDUCE_OPS: set[int] = _e_opcodes(w=W_REDUCE)
"""E opcodes carrying reduction latency, each reading port A alone."""

_M_OCCUPANCY: dict[int, int] = {_OPCODES[name]: cycles for name, cycles in MATRIX_OCCUPANCY.items()}
"""Cycles the matrix controller is busy per M subop, keyed by opcode.

A vmatpush retires on beat acceptance, so its entry is the controller's occupancy rather than its result latency."""

_M_HOLDS_PORT_A: set[int] = {_OPCODES[name] for name in MATRIX_PORT_A}
"""M subops owning VREG read port A while they run, which vmatpop stays out of by never asserting
vreg_read_valid_o."""

_M_RESULT_LATENCY: int = MATRIX_RESULT_LATENCY
"""Cycles from a vmatpush to its first result.

Results leave the array in push order and one vmatpop drains exactly one vmatpush's worth, so the wait is tracked
as a queue where a pop waits for the oldest un-popped push."""

_MXU_OUTPUT_FIFO_DEPTH: int = 32
"""Entries in each lane's MXU output FIFO, mirroring mxu.sv gen_lane_output's OUTPUT_FIFO_DEPTH.

Overflow drops results silently, with no replay or recovery, as vpu_fifo.sv's header states."""

_MXU_ENTRIES_PER_PUSH: int = 4
"""Entries one vmatpush produces, being the NUM_SUBLANES activation rows it streams."""

_MXU_ENTRIES_PER_POP: int = 4
"""Entries one vmatpop consumes across its masked beats."""

_SCHEDULE_CACHE_MAGIC: bytes = b"MTSC1"
"""Header for a remembered schedule, carrying `MTSC` and a format number so a file written by a different layout is
rejected rather than misread."""


@dataclass
class AsmBuilder:
    """Records explicit physical VLIW bundles as a kernel emits them.

    Every slot word returned here is already shifted into its bundle field, and every E-slot word reads its first
    source through VREG port A.

    Attributes:
        bundles: The bundle words recorded so far, in emission order.
    """

    bundles: list[int]

    def __init__(self) -> None:
        self.bundles = []

    def bundle(self, *words: int, **slots: int) -> None:
        """Appends one VLIW bundle assembled from explicitly named slots.

        Args:
            words: One raw bundle word, for a value assembled elsewhere.
            slots: Slot words keyed by slot letter, which are combined into one bundle.

        Raises:
            ValueError: If a bundle follows F:halt or a key names no VLIW slot.
            TypeError: If the call mixes positional and named forms, passes more than one word, or repeats a slot.
        """
        if self.bundles and _is_halt(self.bundles[-1]):
            message = "Unable to append a bundle after F:halt. F:halt must be the final bundle of an image."
            raise ValueError(message)
        if words and slots:
            message = (
                f"Unable to assemble a bundle from {len(words)} positional words and the slots {sorted(slots)}. "
                f"A bundle takes positional words or named slots, but not both."
            )
            raise TypeError(message)
        if len(words) > 1:
            message = (
                f"Unable to assemble one bundle from several words. A bundle takes one word, but got {len(words)}."
            )
            raise TypeError(message)
        if len(set(slots)) != len(slots):
            message = (
                f"Unable to assemble a bundle naming one slot twice. The named slots must be distinct, but got "
                f"{sorted(slots)}."
            )
            raise TypeError(message)
        _check_dma_exclusive(slots)
        word = words[0] if words else 0
        for slot, value in slots.items():
            if slot not in {"e", "v", "m", "x", "s", "d", "f"}:
                message = (
                    f"Unable to assemble a bundle from the given slot. The slot must be one of e, v, m, x, s, d "
                    f"or f, but got {slot!r}."
                )
                raise ValueError(message)
            word |= int(value)
        self.bundles.append(_bundle(word))

    def include_binary(self, path: str | Path) -> None:
        """Inlines an existing 128-bit bundle image into the stream under construction.

        Args:
            path: The bundle image to read.

        Raises:
            ValueError: If the file is empty or its length is not a whole number of bundles.
        """
        data = Path(path).read_bytes()
        if not data or len(data) % BUNDLE_BYTES:
            message = (
                f"Unable to inline the image at {path}. The file must hold a whole number of {BUNDLE_BYTES} byte "
                f"bundles and at least one, but got {len(data)} bytes."
            )
            raise ValueError(message)
        for offset in range(0, len(data), BUNDLE_BYTES):
            self.bundle(int.from_bytes(data[offset : offset + BUNDLE_BYTES], "little"))

    def smov_arg(self, rd: int, arg: int) -> int:
        """Returns an S-slot word moving the launch argument at index `arg` into a scalar register."""
        return _compute_s(rd, arg, op=1)

    def smovi(self, rd: int, value: int) -> int:
        """Returns an S-slot word moving an unsigned immediate into a scalar register."""
        return _compute_s(rd, 0, op=0, imm=value)

    def saddi(self, rd: int, rs: int, value: int) -> int:
        """Returns an S-slot word adding a signed immediate to a scalar register."""
        return _compute_s(rd, rs, op=2, imm=value)

    def vld(self, vreg: int, row: int, *, agu: bool = False, shift: int = 0,
            level: int = 0, stride: int = 1, mask: int = 0xF) -> int:
        """Returns an X-slot word loading NUM_SUBLANES VMEM rows into one VREG.

        Args:
            vreg: Destination VREG.
            row: First VMEM row, before the AGU and SREG terms are added.
            agu: Determines whether the loop induction variable is added to the row.
            shift: Left shift applied to that induction variable.
            level: Which loop's induction variable the access reads, counting 0 as the outermost loop. A level
                naming the wrong loop reads a plausible row and returns wrong numbers rather than faulting.
            stride: Rows between consecutive sublanes. An even stride collides sublanes onto one bank and costs a
                cycle per collision, which `schedule` prices.
            mask: Which of the NUM_SUBLANES sublanes take part.
        """
        return _compute_x(False, vreg, row, agu=agu, shift=shift, level=level,
                          stride=stride, mask=mask)

    def vst(self, vreg: int, row: int, *, agu: bool = False, shift: int = 0,
            level: int = 0, stride: int = 1, mask: int = 0xF) -> int:
        """Returns an X-slot word storing one VREG across NUM_SUBLANES VMEM rows.

        Args:
            vreg: Source VREG.
            row: First VMEM row, before the AGU and SREG terms are added.
            agu: Determines whether the loop induction variable is added to the row.
            shift: Left shift applied to that induction variable.
            level: Which loop's induction variable the access reads, under the `vld` convention.
            stride: Rows between consecutive sublanes, priced as `vld` describes.
            mask: Which of the NUM_SUBLANES sublanes take part.
        """
        return _compute_x(True, vreg, row, agu=agu, shift=shift, level=level,
                          stride=stride, mask=mask)

    def vadd(self, vd: int, va: int, vb: int) -> int:
        """Returns an E-slot word writing the elementwise sum of two VREGs."""
        return _compute_e(1, vd, va, vb)

    def vsub(self, vd: int, va: int, vb: int) -> int:
        """Returns an E-slot word writing the elementwise difference of two VREGs."""
        return _compute_e(2, vd, va, vb)

    def vmul(self, vd: int, va: int, vb: int) -> int:
        """Returns an E-slot word writing the elementwise product of two VREGs."""
        return _compute_e(3, vd, va, vb)

    def vmax(self, vd: int, va: int, vb: int) -> int:
        """Returns an E-slot word writing the elementwise maximum of two VREGs."""
        return _compute_e(11, vd, va, vb)

    def vmin(self, vd: int, va: int, vb: int) -> int:
        """Returns an E-slot word writing the elementwise minimum of two VREGs."""
        return _compute_e(12, vd, va, vb)

    def vtxin(self, va: int, quarter: int) -> int:
        """Returns an E-slot word writing one VREG of rows into the transpose tile.

        Four of these fill a 16x16 tile, where quarter q takes tile rows 4q through 4q+3 from the source VREG's four
        sublanes and writes no VREG. The unit holds the tile until `vtxout` drains it.
        """
        return _compute_e(13, 0, va, quarter)

    def vtxout(self, vd: int, quarter: int) -> int:
        """Returns an E-slot word reading one VREG of tile columns back out.

        Four of these drain the tile, where quarter q writes the destination with tile columns 4q through 4q+3, so
        sublane s of lane l receives tile[l][4q + s].
        """
        return _compute_e(14, vd, 0, quarter)

    def vmov(self, vd: int, va: int) -> int:
        """Returns an E-slot word copying one vector register to another."""
        return _compute_e(4, vd, va, 0)

    def vgelu(self, vd: int, va: int) -> int:
        """Returns an E-slot word activating a register through the SFU's GELU table."""
        return _compute_e(5, vd, va, 0)

    def vexp(self, vd: int, va: int) -> int:
        """Returns an E-slot word exponentiating a register through the SFU table."""
        return _compute_e(6, vd, va, 0)

    def vrecip(self, vd: int, va: int) -> int:
        """Returns an E-slot word reciprocating a register through the SFU table."""
        return _compute_e(7, vd, va, 0)

    def vrsqrt(self, vd: int, va: int) -> int:
        """Returns an E-slot word taking a register's reciprocal square root through the SFU table."""
        return _compute_e(8, vd, va, 0)

    def vredsum(self, vd: int, va: int) -> int:
        """Returns an E-slot word summing one VREG's elements through the reduction tree."""
        return _compute_e(9, vd, va, 0)

    def vredmax(self, vd: int, va: int) -> int:
        """Returns an E-slot word taking the maximum of one VREG's elements through the reduction tree."""
        return _compute_e(10, vd, va, 0)

    def vmatload(self, vreg: int) -> int:
        """Returns an M-slot word parking a weight tile in the matrix array, read from the four VREGs starting at
        `vreg`.
        """
        return _compute_m(1, vreg)

    def vmatpush(self, vreg: int) -> int:
        """Returns an M-slot word streaming one VREG of activations into the matrix array, which reads it for the
        push's whole occupancy.
        """
        return _compute_m(2, vreg)

    def vmatpop(self, vreg: int) -> int:
        """Returns an M-slot word draining one push's worth of results out of the matrix array across its masked
        beats.
        """
        return _compute_m(3, vreg)

    def lbegin(self, hi: int, *, lo: int = 0, step: int = 1) -> int:
        """Returns an F-slot word opening a hardware loop over the half-open range [lo, hi) in steps of `step`, which
        pushes a stack frame.
        """
        return _compute_f(1, (int(lo) << 20) | (int(step) << 16) | int(hi))

    def loop_end(self) -> int:
        """Returns an F-slot word closing the innermost hardware loop."""
        return _compute_f(2, 0)

    def dma_load(self, slot: int, vmem_row: int, rows: int, base_sreg: int, stride_sreg: int) -> int:
        """Returns a complete DMA bundle reading host memory into VMEM.

        The destination `vmem_row` carries the VMEM slot in its top bit, the host base and row stride are read
        from the named scalar registers, and `flush_slot` later waits on `slot`.
        """
        return _dma(False, slot, vmem_row, rows, base_sreg, stride_sreg)

    def dma_store(self, slot: int, vmem_row: int, rows: int, base_sreg: int, stride_sreg: int) -> int:
        """Returns a complete DMA bundle writing VMEM back to host memory.

        The source `vmem_row` carries the VMEM slot in its top bit, the host base and row stride are read from the
        named scalar registers, and `flush_slot` later waits on `slot`.
        """
        return _dma(True, slot, vmem_row, rows, base_sreg, stride_sreg)

    def vmemswap(self) -> int:
        """Returns an F-slot word toggling which VMEM slot the X slot addresses.

        A DMA descriptor names its VMEM slot in an immediate rather than a register, so double buffering a kernel
        works by swapping which slot compute owns rather than by rewriting the descriptor.
        """
        return _compute_f(5, 0)

    def flush_slot(self, mask: int) -> int:
        """Returns an F-slot word blocking until every DMA slot named in `mask` reports done.

        The mask is per DMA slot, so a mask naming a slot that never carried a descriptor waits forever and wedges
        the device until the PL is reprogrammed.
        """
        return _flush(mask)

    def halt(self, word: int = HALT_BUNDLE) -> int:
        """Returns the canonical halt bundle, which must be an image's final word.

        Raises:
            ValueError: If `word` differs from the RTL's canonical halt encoding.
        """
        if _bundle(word) != HALT_BUNDLE:
            message = (
                f"Unable to emit the given word as F:halt. The word must be the canonical RTL halt bundle "
                f"0x{HALT_BUNDLE:x}, but got 0x{int(word):x}."
            )
            raise ValueError(message)
        return HALT_BUNDLE


@dataclass(frozen=True)
class AsmKernel:
    """Holds one kernel's unscheduled bundle stream, which it schedules on demand."""

    name: str
    """The name the kernel's error messages carry."""
    bundles: tuple[int, ...]
    """The bundle words the kernel emitted, in program order."""

    def assemble(self, *, pad_to: int | None = None) -> tuple[int, ...]:
        """Returns the kernel's scheduled image, padded by repeating its final bundle.

        Re-placement stays opt-in through `MINITPU_PACK`, and it forms part of the memo key because it changes the
        image.

        Args:
            pad_to: The bundle count the image is padded to a multiple of, or None to leave it unpadded.

        Returns:
            The scheduled bundle words.
        """
        words = list(_scheduled(self.name, self.bundles, _placement()))
        if pad_to is not None:
            while len(words) % pad_to:
                words.append(words[-1])
        return tuple(words)

    def to_bytes(self, *, pad_to: int | None = None) -> bytes:
        """Serialises the kernel's scheduled image as little-endian bundle words.

        Args:
            pad_to: The bundle count the image is padded to a multiple of, or None to leave it unpadded.

        Returns:
            The image as the bytes an IRAM load expects.
        """
        return b"".join(word.to_bytes(BUNDLE_BYTES, "little") for word in self.assemble(pad_to=pad_to))


def sublane_conflict_cycles(stride: int, mask: int) -> int:
    """Returns the cycles the VMEM compute port is held by one X-slot access.

    Notes:
        vpu_bank_select.sv selects `bank[s] = (row + s*stride) mod NUM_SUBLANES`. Adding the row is a bijection on
        Z/NUM_SUBLANES, so the base rotates which banks are touched while leaving their count alone, and the cost
        follows from the stride and mask alone, both of which are immediates.

    Args:
        stride: Rows between consecutive sublanes.
        mask: Which sublanes take part.

    Returns:
        The maximum number of enabled sublanes landing on one bank, which is 1 when conflict-free.
    """
    banks = [0] * NUM_SUBLANES
    for sublane in range(NUM_SUBLANES):
        if mask >> sublane & 1:
            banks[(sublane * stride) % NUM_SUBLANES] += 1
    return max(banks)


def mxu_output_fifo() -> tuple[int, int, int]:
    """Returns the MXU output FIFO's depth, its entries per vmatpush, and its entries per vmatpop.
    """
    return _MXU_OUTPUT_FIFO_DEPTH, _MXU_ENTRIES_PER_PUSH, _MXU_ENTRIES_PER_POP


def pack(words: list[int]) -> list[int]:
    """Re-places a hand-written bundle stream as densely as the machine allows.

    The builders place every op at an explicit index and pad the gaps with empty bundles, which is legal but leaves
    most of the five compute slots idle. This pass runs ahead of `schedule`, which still derives the exact delays.

    Args:
        words: The emitted bundle stream, before scheduling.

    Returns:
        The packed stream, ready for `schedule`.
    """
    return _reframe(words, _pack_region)


def place(words: list[int]) -> list[int]:
    """Re-places a hand-written bundle stream in program order, one operation per bundle.

    The hand-placed padding inside each region is dropped and `schedule` re-derives the exact delay. Each region is
    then emitted in the order it was written, so the S and M slot placement the machine leaves to the compiler
    survives by construction, at the cost of the co-issue `pack` reaches. `tools/check_pack.py` measures that cost.

    Args:
        words: The emitted bundle stream, before scheduling.

    Returns:
        The re-placed stream, ready for `schedule`.
    """
    return _reframe(words, list)


def schedule(words: list[int]) -> list[int]:
    """Returns one bundle stream with its delay fields filled in, so the schedule is legal.

    Delays are filled in place and filler bundles are inserted where a wait exceeds `MAX_DELAY`. This pass keeps
    the stream in order, so it leaves SREG WAW and WAR and VMEM ownership to the caller.

    Args:
        words: The bundle stream to space, in program order.

    Returns:
        The stream carrying legal delays, which is longer than the input where fillers were needed.

    Raises:
        ValueError: If a bundle co-issues two port-A consumers, a vmatpush overflows the MXU output FIFO, or the
            delay fixpoint fails to converge.
    """
    _check_port_a_exclusive(words=words)
    _check_matrix_fifo(words=words)
    _check_flush_slots(words=words)
    out = list(words)
    for _ in range(256):
        ready: dict[int, int] = {}
        s_ready: dict[int, int] = {}
        wb: set[int] = set()
        result_ready: list[int] = []   # oldest un-popped push first
        matrix_free = port_a_free = vmem_free = 0
        t_prev = 0
        insert_at = None
        for i, word in enumerate(out):
            d = _decode(word)
            t = 0 if i == 0 else t_prev + 1 + _delay_of(out[i - 1])
            # Prices SREG RAW for every bundle, DMA descriptors included, since a descriptor reads its base and
            # stride from scalar registers. WAW and WAR come free from the in-order advance of t.
            for register in d["s_reads"]:
                t = max(t, s_ready.get(register, 0))
            if not d["dma"]:
                for v in d["reads"]:
                    t = max(t, ready.get(v, 0))
                if d["m_op"]:
                    t = max(t, matrix_free)
                if d["m_op"] == 3 and result_ready:
                    t = max(t, result_ready[0])
                if d["port_a"]:
                    t = max(t, port_a_free)
                if d["vmem_cycles"]:
                    t = max(t, vmem_free)
                if d["writes"]:
                    span = _wb_span(d)
                    while any(t + w in wb for w in span):
                        t += 1
            if i > 0 and t - t_prev - 1 > MAX_DELAY:
                gap = t - t_prev - 1
                fillers = 0
                while (fillers + 1) * MAX_DELAY + fillers < gap:
                    fillers += 1
                insert_at = (i, fillers)
                break
            if not d["dma"]:
                for vreg, w in d["writes"]:
                    wb.update(t + s for s in _wb_span_of(d, w))
                    ready[vreg] = t + max(_wb_span_of(d, w)) + 1
                if d["vmem_cycles"]:
                    vmem_free = t + d["vmem_cycles"]
                if d["m_op"]:
                    matrix_free = t + _M_OCCUPANCY[d["m_op"]] + 1
                    if d["m_op"] in _M_HOLDS_PORT_A:
                        port_a_free = matrix_free
                    if d["m_op"] == 2:
                        result_ready.append(t + _M_RESULT_LATENCY)
                    elif d["m_op"] == 3 and result_ready:
                        result_ready.pop(0)
            for register, latency in d["s_writes"]:
                s_ready[register] = t + latency + 1
            if i > 0:
                out[i - 1] = _set_delay(
                    out[i - 1], max(_delay_of(out[i - 1]), t - t_prev - 1))
            t_prev = t
        if insert_at is None:
            return out
        where, fillers = insert_at
        for _ in range(fillers):
            out.insert(where, _set_delay(0, MAX_DELAY))
    message = (
        f"Unable to space the given bundle stream. Filler insertion must reach a fixpoint within 256 passes, but "
        f"the stream of {len(words)} bundles still needed one."
    )
    raise ValueError(message)


def analyze_stalls(words: list[int]) -> dict:
    """Attributes every cycle a scheduled image spends waiting to the constraint that caused it.

    Args:
        words: A scheduled image, as `schedule` returns.

    Returns:
        The `cycles` the image occupies, the `bundles` it contains, and `stalls`, mapping each cause to the cycles
        charged to it.
    """
    causes = {"operand": 0, "matrix_busy": 0, "matrix_result": 0, "port_a": 0, "write_port": 0}
    ready: dict[int, int] = {}
    wb: set[int] = set()
    result_ready: list[int] = []
    matrix_free = port_a_free = 0
    t_prev = 0
    for index, word in enumerate(words):
        d = _decode(word)
        base = 0 if index == 0 else t_prev + 1
        t = base
        if not d["dma"]:
            wants = {"operand": max((ready.get(v, 0) for v in d["reads"]), default=0)}
            if d["m_op"]:
                wants["matrix_busy"] = matrix_free
            if d["m_op"] == 3 and result_ready:
                wants["matrix_result"] = result_ready[0]
            if d["port_a"]:
                wants["port_a"] = port_a_free
            t = max(base, *wants.values())
            if t > base:
                causes[max(wants, key=lambda cause: wants[cause])] += t - base
            if d["writes"]:
                span = _wb_span(d)
                blocked = t
                while any(t + w in wb for w in span):
                    t += 1
                causes["write_port"] += t - blocked
            for vreg, w in d["writes"]:
                wb.update(t + offset for offset in _wb_span_of(d, w))
                ready[vreg] = t + max(_wb_span_of(d, w)) + 1
            if d["m_op"]:
                matrix_free = t + _M_OCCUPANCY[d["m_op"]] + 1
                if d["m_op"] in _M_HOLDS_PORT_A:
                    port_a_free = matrix_free
                if d["m_op"] == 2:
                    result_ready.append(t + _M_RESULT_LATENCY)
                elif d["m_op"] == 3 and result_ready:
                    result_ready.pop(0)
        t_prev = t
    return {"cycles": t_prev + 1, "bundles": len(words), "stalls": causes}


def asm(function: Callable[[], None], *, namespace: str = "asm") -> AsmKernel:
    """Captures the bundles an explicit VLIW construction function emits.

    A fresh builder is bound into the function's own globals under `namespace` for the call and the previous
    binding is restored afterwards, so a kernel body names the builder without taking it as a parameter. The
    kernel is assembled once here, so a malformed one fails where it is written.

    Args:
        function: The kernel body, which emits its bundles through the bound builder.
        namespace: The global name the builder is bound to for the call.

    Returns:
        The captured kernel.

    Raises:
        ValueError: If the emitted stream is empty or ends in something other than a halt.
    """
    builder = AsmBuilder()
    globals_dict = function.__globals__
    previous = globals_dict.get(namespace, None)
    globals_dict[namespace] = builder
    try:
        function()
    finally:
        if previous is None:
            globals_dict.pop(namespace, None)
        else:
            globals_dict[namespace] = previous
    kernel = AsmKernel(function.__name__, tuple(builder.bundles))
    kernel.assemble()
    return kernel


def _bundle(value: int) -> int:
    """Validates that one assembled word fits a physical bundle.

    Args:
        value: The assembled bundle word.

    Returns:
        The bundle word.

    Raises:
        ValueError: If the word does not fit BUNDLE_BITS bits.
    """
    value = int(value)
    if not 0 <= value < (1 << BUNDLE_BITS):
        message = (
            f"Unable to encode the given bundle word. The word must fit in {BUNDLE_BITS} bits, but got 0x{value:x}."
        )
        raise ValueError(message)
    return value


def _is_halt(word: int) -> bool:
    """Determines whether one bundle word carries the canonical F:halt encoding."""
    return word == HALT_BUNDLE


def _check_dma_exclusive(slots: dict[str, int]) -> None:
    """Rejects a DMA bundle that also names a compute slot.

    Args:
        slots: The named slots one bundle assembles, keyed by slot letter.

    Raises:
        ValueError: If the bundle names the D slot together with any compute slot.
    """
    if "d" not in slots:
        return
    conflicting = sorted(DMA_EXCLUSIVE_SLOTS & set(slots))
    if conflicting:
        message = (
            f"Unable to assemble a D slot co-issued with {conflicting}. The D slot is exclusive, because a DMA "
            f"bundle encodes a descriptor over the bits a compute bundle spends on those slots. The overlap "
            f"corrupts the descriptor or drops the slot, silently in both cases. Emit the DMA in its own bundle."
        )
        raise ValueError(message)


def _compute_s(rd: int, rs: int, *, op: int, imm: int = 0) -> int:
    """Returns the S-slot word for one scalar operation, shifted into its field.

    scalar_agu.sv sign-extends the immediate field for SADDI and zero-extends it for SMOVI, so the accepted range
    spans both. Masking the immediate is what lets SADDI encode a negative addend. Without it, a negative immediate
    corrupts the word and an immediate above 0xFFFF overflows into the rs field, silently in both cases.

    Args:
        rd: Destination scalar register.
        rs: Source scalar register.
        op: Scalar opcode, being 0 for SMOVI, 1 for SMOV_ARG and 2 for SADDI.
        imm: Immediate operand, read signed by SADDI and unsigned by SMOVI.

    Returns:
        The S-slot word, shifted into its bundle field.

    Raises:
        ValueError: If the immediate reaches outside the range both readings span.
    """
    if not -0x8000 <= int(imm) <= 0xFFFF:
        message = (
            f"Unable to encode the given scalar immediate. The field is 16 bits, which SADDI reads signed "
            f"(-32768..32767) and SMOVI reads unsigned (0..65535), but got {imm}."
        )
        raise ValueError(message)
    s = ((1 << 26) | (int(op) << 23) | (int(rd) << 21) | (int(rs) << 19)
         | (int(imm) & 0xFFFF))
    return s << _S_SHIFT


def _compute_e(op: int, rd: int, ra: int, rb: int) -> int:
    """Returns the E-slot word for one vector operation, where the unary opcodes hold `rb` at zero."""
    e = (int(op) << 15) | (int(rd) << 10) | (int(ra) << 5) | int(rb)
    return e << _E_SHIFT


def _compute_m(op: int, reg: int) -> int:
    """Returns the M-slot word for one matrix operation, keyed 1 for vmatload, 2 for vmatpush and 3 for vmatpop."""
    return ((int(op) << 5) | int(reg)) << _M_SHIFT


def _compute_x(store: bool, vreg: int, row: int, *, agu: bool, shift: int,
               level: int = 0, stride: int = 1, mask: int = 0xF) -> int:
    """Returns the X-slot word for one VMEM access, shifted into its field.

    Notes:
        The 40-bit x_slot_t at NUM_SUBLANES=4, ROW_ADDR_W=13 and SUBLANE_STRIDE_W=7 is laid out as valid[39],
        op[38], vreg_idx[37:33], literal[32:20], agu_valid[19], agu_level[18:17], agu_shift[16:14], sreg_valid[13],
        sreg_idx[12:11], sublane_stride[10:4] and sublane_mask[3:0]. The literal occupies 13 bits growing upward
        from bit 20, so widening it moves valid, op and vreg_idx alone.

    Args:
        store: Determines whether the access writes VMEM rather than reading it.
        vreg: The VREG the access writes or reads.
        row: First VMEM row, before the AGU and SREG terms are added.
        agu: Determines whether the loop induction variable is added to the row.
        shift: Left shift applied to that induction variable.
        level: Which loop's induction variable the access reads, counting 0 as the outermost loop.
        stride: Rows between consecutive sublanes.
        mask: Which sublanes take part.

    Returns:
        The X-slot word, shifted into its bundle field.

    Raises:
        ValueError: If the stride overflows its field, the mask names no sublane, or the level overflows the
            loop stack.
    """
    if not 0 <= stride < (1 << SUBLANE_STRIDE_W):
        message = (
            f"Unable to encode the given sublane stride. The field is {SUBLANE_STRIDE_W} bits, so the stride must "
            f"be under {1 << SUBLANE_STRIDE_W}, but got {stride}."
        )
        raise ValueError(message)
    if not 0 < mask < (1 << NUM_SUBLANES):
        message = (
            f"Unable to encode the given sublane mask. The mask must name at least one of the {NUM_SUBLANES} "
            f"sublanes, but got 0x{mask:x}."
        )
        raise ValueError(message)
    x = (1 << 39) | (int(store) << 38) | (int(vreg) << 33) | (int(row) << 20)
    if not 0 <= level < (1 << LEVEL_SEL_W):
        message = (
            f"Unable to encode the given AGU level. The field is {LEVEL_SEL_W} bits, so the loop stack is "
            f"{1 << LEVEL_SEL_W} deep with level 0 the outermost loop, but got {level}."
        )
        raise ValueError(message)
    x |= int(agu) << 19 | int(level) << 17 | int(shift) << 14 | (int(stride) << 4) | int(mask)
    return x << _X_SHIFT


def _compute_f(op: int, payload: int) -> int:
    """Returns the F-slot word for one control operation, keyed 1 for loop begin, 2 for loop end and 5 for
    vmemswap.
    """
    return (int(op) << (_F_SHIFT + F_SLOT_W - 3)) | (int(payload) << _F_SHIFT)


def _dma(store: bool, slot: int, vmem_row: int, rows: int, base_sreg: int, stride_sreg: int) -> int:
    """Returns a complete DMA bundle carrying one descriptor.

    Notes:
        The 36-bit d_wire_t at ROW_ADDR_W=13, where vmem_row is the 14-bit concatenation of the slot bit and the
        row, is laid out as is_store[35], slot_sel[34], vmem_row[33:20], src_mode[19], pool_idx[18:11], rows[10:4],
        base_sreg[3:2] and stride_sreg[1:0]. src_mode is held at D_SRC_SREG, so the base and stride are read from
        scalar registers.

    Args:
        store: Determines whether the descriptor writes host memory rather than reading it.
        slot: Which DMA slot carries the descriptor.
        vmem_row: The VMEM row, whose top bit selects the VMEM slot.
        rows: Number of rows the descriptor transfers.
        base_sreg: Scalar register holding the host base address.
        stride_sreg: Scalar register holding the host row stride.

    Returns:
        The complete DMA bundle word.
    """
    d = (int(store) << 35) | (int(slot) << 34) | (int(vmem_row) << 20)
    d |= (1 << 19) | ((int(rows) - 1) << 4) | (int(base_sreg) << 2) | int(stride_sreg)
    return (1 << 127) | (d << 91)


def _flush(mask: int) -> int:
    """Returns the F-slot word blocking until every DMA slot named in `mask` reports done."""
    return (3 << (_F_SHIFT + F_SLOT_W - 3)) | (int(mask) << _F_SHIFT)


def _decode(word: int) -> dict:
    """Unpacks the physical fields a scheduler needs from one 128-bit bundle.

    Only VREG and SREG hazards are modelled here, so a DMA descriptor reports the scalar registers it reads and no
    VREG traffic.

    Args:
        word: The bundle word to decode.

    Returns:
        A mapping carrying `dma`, the VREG registers the bundle `reads`, its `writes` as register and writeback
        offset pairs, the `vmem_cycles` an X access holds the port for, the `m_op` subop and its `m_reg`, whether
        the bundle claims `port_a`, the `f_op` control opcode, and the `s_reads` and `s_writes` scalar registers.
    """
    if (word >> 127) & 1:
        # Takes its base and stride from scalar registers, with nothing interlocking that read against the write
        # that set them. `_dma` holds the field positions.
        return {"dma": True, "reads": set(), "writes": [], "vmem_cycles": 0, "m_op": 0, "m_reg": 0,
                "port_a": False, "f_op": 0,
                "s_reads": {(word >> 93) & 0x3, (word >> 91) & 0x3}, "s_writes": [],
                "dma_slot": (word >> 125) & 1, "flush_mask": 0}
    e = (word >> _E_SHIFT) & ((1 << E_SLOT_W) - 1)
    m = (word >> _M_SHIFT) & ((1 << M_SLOT_W) - 1)
    x = (word >> _X_SHIFT) & ((1 << X_SLOT_W) - 1)
    f = (word >> _F_SHIFT) & ((1 << F_SLOT_W) - 1)
    s = (word >> _S_SHIFT) & ((1 << S_SLOT_W) - 1)
    d = {"dma": False, "reads": set(), "writes": [], "vmem_cycles": 0,
         "m_op": (m >> 5) & 0x3 if m else 0, "m_reg": m & 0x1F,
         "port_a": False, "f_op": (f >> 24) & 0x7 if f else 0,
         "s_reads": set(), "s_writes": [], "dma_slot": None,
         "flush_mask": f & 0x3 if f and (f >> 24) & 0x7 == _F_FLUSH else 0}
    if s:
        # Decodes valid[26] op[25:23] rd[22:21] rs[20:19] imm[15:0], as `_compute_s` encodes it. SMOVI and
        # SMOV_ARG leave rs unread, and claiming that read anyway costs a scheduling gap alone.
        d["s_reads"].add((s >> 19) & 0x3)
        d["s_writes"].append((((s >> 21) & 0x3), W_SCALAR))
    if e:
        op, rd, ra, rb = (e >> 15) & 0xF, (e >> 10) & 0x1F, (e >> 5) & 0x1F, e & 0x1F
        d["reads"].add(ra)
        d["port_a"] = True
        if op in _E_ALU_OPS:
            if op in _E_BINARY_OPS:
                d["reads"].add(rb)
            d["writes"].append((rd, W_ALU))
        elif op in _E_SFU_OPS:
            d["writes"].append((rd, W_SFU))
        elif op in _E_REDUCE_OPS:
            d["writes"].append((rd, W_REDUCE))
        elif op == _E_TXOUT:
            # Reads the staging tile rather than a VREG, so the port-A read recorded above is spurious. Keeping
            # it costs a scheduling gap alone, which is the safe direction here.
            d["writes"].append((rd, W_TRANSPOSE))
    if x:
        store, vreg = (x >> 38) & 1, (x >> 33) & 0x1F
        # Carries one bit per sublane, so sublane_mask is NUM_SUBLANES bits wide.
        d["vmem_cycles"] = sublane_conflict_cycles((x >> 4) & ((1 << SUBLANE_STRIDE_W) - 1),
                                                   x & ((1 << NUM_SUBLANES) - 1))
        if store:
            d["reads"].add(vreg)
            d["port_a"] = True
        else:
            # Merges a split load one pass per cycle, so its writeback lands (degree - 1) cycles later than a
            # conflict-free one. Measured against the RTL, a 4-way conflicting vld's consumer is legal at 10.
            d["writes"].append((vreg, W_VLD + d["vmem_cycles"] - 1))
    if d["m_op"] == 1:
        # Reads NUM_LANES/NUM_SUBLANES consecutive registers.
        d["reads"].update((d["m_reg"] + r) & 0x1F for r in range(4))
        d["port_a"] = True
    elif d["m_op"] == 2:
        d["reads"].add(d["m_reg"])
        d["port_a"] = True
    elif d["m_op"] == 3:
        # Writes across the masked beats, which W_MPOP_LAST closes.
        d["writes"].append((d["m_reg"], W_MPOP_LAST))
    return d


def _port_a_readers(word: int) -> list[str]:
    """Names every VREG read-port-A consumer one bundle issues, empty for a DMA bundle.

    vpu.sv's `effective_raddr_a` is a single physical port behind a static-priority mux the matrix controller wins,
    so two consumers in one bundle issue together and the loser silently reads the winner's address. A vmatpop and
    a vld stay out of this, reading the array's own output and VMEM.
    """
    if (word >> 127) & 1:
        return []
    readers = []
    if (word >> _E_SHIFT) & ((1 << E_SLOT_W) - 1):
        readers.append("E")
    x = (word >> _X_SHIFT) & ((1 << X_SLOT_W) - 1)
    if x and (x >> 38) & 1:
        readers.append("X:vst")
    m = (word >> _M_SHIFT) & ((1 << M_SLOT_W) - 1)
    m_op = (m >> 5) & 0x3 if m else 0
    if m_op == 1:
        readers.append("M:vmatload")
    elif m_op == 2:
        readers.append("M:vmatpush")
    return readers


def _slot_mask(word: int) -> int:
    """Returns one bit per occupied E, M, X, S and F slot, from bit 0, so two bundles are testable as disjoint."""
    mask = 0
    for bit, (shift, width) in enumerate(((_E_SHIFT, E_SLOT_W), (_M_SHIFT, M_SLOT_W),
                                          (_X_SHIFT, X_SLOT_W), (_S_SHIFT, S_SLOT_W),
                                          (_F_SHIFT, F_SLOT_W))):
        if (word >> shift) & ((1 << width) - 1):
            mask |= 1 << bit
    return mask


def _is_barrier(word: int) -> bool:
    """Determines whether a bundle must keep its emitted position exactly.

    A DMA bundle is a barrier, because a descriptor's completion is observed through `flush_slot` and VMEM aliasing
    against a `vld` is beyond what this pass proves. So is any F slot, which preserves control flow bit for bit. M
    and S slots are barriers because their consumers sit outside the region: a descriptor reads the SREG an S op
    writes across the barrier that ends it, and `_pack_region` prices the uninterlocked scalar writeback only
    against the ops it holds.
    """
    if (word >> 127) & 1:
        return True
    if (word >> _F_SHIFT) & ((1 << F_SLOT_W) - 1):
        return True
    if (word >> _M_SHIFT) & ((1 << M_SLOT_W) - 1):
        return True
    return bool((word >> _S_SHIFT) & ((1 << S_SLOT_W) - 1))


def _wb_span_of(decoded: dict, w: int) -> range:
    """Returns the offsets, relative to issue, one writeback holds the port for, widened over a vmatpop's beats."""
    if decoded["m_op"] == 3 and w == W_MPOP_LAST:
        return range(W_MPOP_FIRST, W_MPOP_LAST + 1)
    return range(w, w + 1)


def _wb_span(decoded: dict) -> list[int]:
    """Returns every offset, relative to issue, one bundle holds a writeback port for."""
    return [s for _, w in decoded["writes"] for s in _wb_span_of(decoded, w)]


def _delay_of(word: int) -> int:
    """Reads the cycles a bundle holds the fetch head for, honouring the two bundle formats."""
    shift = _DMA_DELAY_SHIFT if (word >> 127) & 1 else _DELAY_SHIFT
    return (word >> shift) & ((1 << DELAY_W) - 1)


def _set_delay(word: int, cycles: int) -> int:
    """Replaces a bundle's delay field, honouring the two bundle formats."""
    shift = _DMA_DELAY_SHIFT if (word >> 127) & 1 else _DELAY_SHIFT
    return (word & ~(((1 << DELAY_W) - 1) << shift)) | (cycles << shift)


def _check_matrix_fifo(words: list[int]) -> None:
    """Rejects a schedule able to overflow the MXU's per-lane output FIFO.

    Raises:
        ValueError: If a vmatpush leaves more results outstanding than one lane's FIFO holds.
    """
    outstanding = 0
    peak_index = None
    for index, word in enumerate(words):
        if (word >> 127) & 1:
            continue
        m = (word >> _M_SHIFT) & ((1 << M_SLOT_W) - 1)
        m_op = (m >> 5) & 0x3 if m else 0
        if m_op == 2:
            outstanding += _MXU_ENTRIES_PER_PUSH
            if outstanding > _MXU_OUTPUT_FIFO_DEPTH and peak_index is None:
                peak_index = index
        elif m_op == 3:
            outstanding = max(0, outstanding - _MXU_ENTRIES_PER_POP)
    if peak_index is not None:
        pushes = _MXU_OUTPUT_FIFO_DEPTH // _MXU_ENTRIES_PER_PUSH
        message = (
            f"Unable to schedule a stream whose bundle {peak_index} overflows the MXU output FIFO. Each lane's "
            f"FIFO holds exactly {_MXU_OUTPUT_FIFO_DEPTH} results, and that vmatpush would leave more than that "
            f"outstanding. At {_MXU_ENTRIES_PER_PUSH} entries per vmatpush this caps a burst at {pushes} pushes "
            f"before a vmatpop drains it. Hardware drops the results silently, with no replay, so each group must "
            f"be drained before the next is pushed."
        )
        raise ValueError(message)


def _check_flush_slots(words: list[int]) -> None:
    """Rejects a stream flushing a DMA slot no descriptor in it ever targets.

    `flush_slot` blocks until `dma_slot_done` covers every bit of its mask and only a descriptor
    re-asserts that bit, so waiting on an unused slot never returns and the device stays wedged with no
    timeout and no error. Slots are counted over the whole stream rather than per flush, because a loop
    body's flush legitimately collects a descriptor issued in the previous iteration.

    Raises:
        ValueError: If a flush names a DMA slot the stream issues no descriptor on.
    """
    issued = {d["dma_slot"] for d in map(_decode, words) if d["dma"]}
    for index, word in enumerate(words):
        waited = _decode(word)["flush_mask"]
        orphans = {slot for slot in range(DMA_SLOT_COUNT) if waited >> slot & 1} - issued
        if orphans:
            message = (
                f"Unable to schedule bundle {index}, whose flush_slot waits on DMA "
                f"{'slots' if len(orphans) > 1 else 'slot'} {sorted(orphans)}. No descriptor in this "
                f"stream targets {'them' if len(orphans) > 1 else 'it'}, and only a descriptor asserts "
                f"the done bit the flush waits for, so the flush never returns and the device stays "
                f"wedged until the PL is reprogrammed. Flush the mask the kernel actually issued."
            )
            raise ValueError(message)


def _check_port_a_exclusive(words: list[int]) -> None:
    """Rejects a bundle naming more than one VREG read-port-A consumer.

    Raises:
        ValueError: If one bundle co-issues two consumers of VREG read port A.
    """
    for index, word in enumerate(words):
        readers = _port_a_readers(word)
        if len(readers) > 1:
            message = (
                f"Unable to schedule bundle {index}, which co-issues {readers}. Those all read VREG port A, a "
                f"single physical read behind a static-priority mux the matrix controller wins, so the losing "
                f"consumer silently reads the winner's register. No interlock covers it and no delay repairs it, "
                f"so the consumers must be split across bundles."
            )
            raise ValueError(message)


def _pack_region(ops: list[int]) -> list[int]:
    """List-schedules one region, merging independent ops into shared bundles.

    Every op is placed at the earliest cycle its dependences and the slot resources allow, in program order, so an
    op stays behind everything it depends on. Merging is a bitwise OR, which holds because no word inside a region
    carries the DMA bit and the delay fields are still zero, so the slots stay disjoint.

    Args:
        ops: The region's non-empty bundles, in program order.

    Returns:
        The same ops merged and reordered, still in cycle order.
    """
    if len(ops) < 2:
        return list(ops)
    decoded = [_decode(word) for word in ops]
    cycle_of: list[int] = [0] * len(ops)
    # Holds each cycle's merged word, the slot mask it occupies, and its port-A consumer count.
    slots: dict[int, list] = {}
    pushes: list[int] = []
    pops = 0

    for index, (word, this) in enumerate(zip(ops, decoded)):
        ready = 0
        this_writes = {register for register, _ in this["writes"]}
        this_s_writes = {register for register, _ in this["s_writes"]}
        for earlier in range(index):
            that, at = decoded[earlier], cycle_of[earlier]
            gap = 0
            occupancy = _M_OCCUPANCY.get(that["m_op"], 0)
            for register, latency in that["writes"]:
                if register in this["reads"]:
                    gap = max(gap, latency + 1)
                if register in this_writes:
                    gap = max(gap, 1)
            for register, latency in that["s_writes"]:
                if register in this["s_reads"]:
                    gap = max(gap, latency + 1)
                if register in this_s_writes:
                    gap = max(gap, 1)
            if any(r in that["reads"] for r in this_writes):
                # Holds the operand for the streaming op's whole occupancy.
                gap = max(gap, occupancy or 1)
            if any(r in that["s_reads"] for r in this_s_writes):
                gap = max(gap, 1)
            if that["m_op"] and this["m_op"]:
                gap = max(gap, occupancy)
            if that["m_op"] == 1 and this["port_a"]:
                # Owns port A for the whole of the vmatload's occupancy.
                gap = max(gap, occupancy)
            if that["vmem_cycles"] and this["vmem_cycles"]:
                if (((word >> _X_SHIFT) >> 38) & 1
                        or ((ops[earlier] >> _X_SHIFT) >> 38) & 1):
                    gap = max(gap, 1)
            if gap:
                ready = max(ready, at + gap)

        if this["m_op"] == 3 and pops < len(pushes):
            # Drains the i-th push with the i-th pop, which waits out the array's result latency.
            ready = max(ready, cycle_of[pushes[pops]] + _M_RESULT_LATENCY)
            pops += 1
        if this["m_op"] == 2:
            pushes.append(index)

        mask, port_a = _slot_mask(word), len(_port_a_readers(word))
        while True:
            held = slots.get(ready)
            if held is None or (not (held[1] & mask) and held[2] + port_a <= 1):
                break
            ready += 1
        held = slots.setdefault(ready, [0, 0, 0])
        held[0] |= word
        held[1] |= mask
        held[2] += port_a
        cycle_of[index] = ready

    return [slots[cycle][0] for cycle in sorted(slots)]


def _reframe(words: list[int], region_placer: Callable[[list[int]], list[int]]) -> list[int]:
    """Splits a stream at its barriers and hands each region to `region_placer`.

    Empty bundles inside a region are dropped, because `schedule` re-derives the delay. Empty bundles immediately
    before a barrier are re-emitted, because a DMA descriptor reads its base and stride from scalar registers
    across that boundary and the gap is the uninterlocked scalar writeback.

    Args:
        words: The emitted bundle stream, before scheduling.
        region_placer: The pass applied to each region between barriers.

    Returns:
        The re-placed stream, with every barrier back at its own position.
    """
    out: list[int] = []
    region: list[int] = []
    pending_nops = 0
    for word in words:
        if _is_barrier(word):
            out.extend(region_placer(region))
            region = []
            out.extend([0] * pending_nops)
            pending_nops = 0
            out.append(word)
        elif word:
            region.append(word)
        elif region:
            pending_nops = 0
        else:
            pending_nops += 1
    out.extend(region_placer(region))
    out.extend([0] * pending_nops)
    return out


def _schedule_cache_dir() -> Path | None:
    """Returns the directory where schedules are remembered between processes, or None when the memo is off.
    """
    setting = os.environ.get("MINITPU_SCHEDULE_CACHE")
    if setting == "off":
        return None
    if setting:
        return Path(setting)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "minitpu"


@functools.lru_cache(maxsize=1)
def _scheduler_digest() -> bytes:
    """Digests this module's own source, so a scheduler change invalidates the memo.

    Without it, editing `schedule` would leave every remembered image stale, which this machine hides: it carries
    no interlock and no bounds check, and returns plausible numbers.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).digest()


def _schedule_digest(bundles: tuple[int, ...], placement: str) -> str:
    """Names the remembered schedule for one bundle stream under one re-placement pass."""
    accumulator = hashlib.sha256()
    accumulator.update(_scheduler_digest())
    accumulator.update(placement.encode())
    accumulator.update(b"\x00")
    accumulator.update(b"".join(
        word.to_bytes(BUNDLE_BYTES, "little") for word in bundles))
    return accumulator.hexdigest()


def _remembered_schedule(digest: str) -> tuple[int, ...] | None:
    """Reads back a remembered schedule, or None when the memo is off, absent or unsound.
    """
    directory = _schedule_cache_dir()
    if directory is None:
        return None
    try:
        blob = (directory / f"{digest}.sched").read_bytes()
    except OSError:
        return None
    head = len(_SCHEDULE_CACHE_MAGIC) + 32
    if len(blob) < head or not blob.startswith(_SCHEDULE_CACHE_MAGIC):
        return None
    payload = blob[head:]
    if len(payload) % BUNDLE_BYTES or hashlib.sha256(payload).digest() != blob[len(_SCHEDULE_CACHE_MAGIC):head]:
        return None
    return tuple(int.from_bytes(payload[at:at + BUNDLE_BYTES], "little")
                 for at in range(0, len(payload), BUNDLE_BYTES))


def _remember_schedule(digest: str, words: tuple[int, ...]) -> None:
    """Writes a schedule down for the next process, atomically or not at all."""
    directory = _schedule_cache_dir()
    if directory is None:
        return
    payload = b"".join(word.to_bytes(BUNDLE_BYTES, "little") for word in words)
    blob = _SCHEDULE_CACHE_MAGIC + hashlib.sha256(payload).digest() + payload
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Renames within the same directory, so every reader sees a whole file and a crashed writer leaves a
        # temporary rather than a plausible schedule.
        temporary = directory / f"{digest}.{os.getpid()}"
        temporary.write_bytes(blob)
        os.replace(temporary, directory / f"{digest}.sched")
    except OSError:
        return


_PLACEMENTS: dict[str, Callable[[list[int]], list[int]] | None] = {
    "": None, "place": place, "pack": pack,
}
"""Re-placement passes `MINITPU_PACK` selects between, where "" keeps the builder's own hand placement."""


def _placement() -> str:
    """Reads the re-placement pass `MINITPU_PACK` selects, where "1" is an accepted spelling of "pack"."""
    selected = os.environ.get("MINITPU_PACK", "")
    return "pack" if selected == "1" else selected


@functools.lru_cache(maxsize=None)
def _scheduled(name: str, bundles: tuple[int, ...], placement: str) -> tuple[int, ...]:
    """Schedules one bundle stream, once per distinct stream in a process and once per machine on disk.

    Args:
        name: The kernel's name, for diagnostics.
        bundles: The unscheduled bundle stream.
        placement: Which re-placement pass `MINITPU_PACK` selected, being "pack" for the reordering one, "place"
            for the order-preserving one, and "" to keep the builder's own hand placement.

    Returns:
        The scheduled words, unpadded.

    Raises:
        ValueError: If the stream is empty, ends in something other than a halt, or `placement` names no pass.
    """
    if not bundles:
        message = f"Unable to schedule {name}. A kernel must carry at least one bundle, but its stream is empty."
        raise ValueError(message)
    if not _is_halt(bundles[-1]):
        message = (
            f"Unable to schedule {name}. The final bundle of an image must carry F:halt, but got "
            f"0x{bundles[-1]:x}."
        )
        raise ValueError(message)
    if placement not in _PLACEMENTS:
        message = (
            f"Unable to schedule {name}. MINITPU_PACK must name one of "
            f"{', '.join(repr(key) for key in sorted(_PLACEMENTS))}, but got {placement!r}."
        )
        raise ValueError(message)
    digest = _schedule_digest(bundles, placement)
    words = _remembered_schedule(digest)
    if words is None:
        stream = list(bundles)
        if _PLACEMENTS[placement] is not None:
            stream = _PLACEMENTS[placement](stream)
        words = tuple(schedule(stream))
        _remember_schedule(digest, words)
    return words


__all__ = ["AsmBuilder", "AsmKernel", "BUNDLE_BITS", "BUNDLE_BYTES", "LEVEL_SEL_W", "SUBLANE_STRIDE_W",
           "analyze_stalls", "asm", "mxu_output_fifo", "pack", "place", "schedule",
           "sublane_conflict_cycles"]
