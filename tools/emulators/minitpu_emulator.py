#!/usr/bin/env python3
"""Executes a MiniTPU image functionally, so a compiled kernel's numbers can be checked without the board.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import deque

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from codegen import asm  # noqa: E402

_LANES: int = 16
"""Lanes one VMEM row and one vector register sublane span."""

_SUBLANES: int = asm.NUM_SUBLANES
"""Sublanes one vector register holds, sublane i reading the access's row plus i."""

_ROW_BYTES: int = _LANES * 2
"""Bytes one VMEM row moves to or from DRAM, being sixteen BF16 elements."""

_VMEM_ROWS: int = 8192
"""Rows one VMEM bank holds, which an access's row wraps within."""

_BANK_SHIFT: int = 13
"""Bit of a descriptor's VMEM address selecting the physical bank."""

_MXU_FIFO_ENTRIES: int = 32
"""Entries each lane's MXU output FIFO holds, a push producing one per sublane."""

_STEP_LIMIT: int = 100_000_000
"""Bundles an image may run before it is taken to have no reachable halt."""

_ERF = np.vectorize(math.erf, otypes=[np.float64])
"""Applies the error function elementwise, which the GELU is defined by."""


class EmulationError(RuntimeError):
    """Signals an image doing something the machine would not survive or cannot express."""


def to_bf16(values: NDArray) -> NDArray[np.float32]:
    """Rounds float values to the nearest BF16, ties to even, as the machine's writeback does."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def _field(word: int, shift: int, width: int) -> int:
    """Returns the `width`-bit field of `word` starting at bit `shift`."""
    return (word >> shift) & ((1 << width) - 1)


class Machine:
    """Holds one launch's machine state: DRAM, VMEM, the register files, the MXU and the transpose array.

    Args:
        dram: DRAM as bytes, addressed exactly as the image's scalar registers address it, which the machine shares
            rather than copies so DRAM persists across launches as the board's does.
        arguments: The launch arguments the image reads with smov_arg, each a DRAM byte address or a scalar.

    Attributes:
        dram: DRAM, as bytes.
        arguments: The launch arguments.
        vmem: Both VMEM banks, as float32 rows of sixteen lanes.
        compute_bank: The bank vld and vst address, which vmemswap exchanges.
        vregs: The vector registers, each four sublanes of sixteen lanes.
        sregs: The scalar registers.
        weights: The weight tile the last vmatload loaded, or None before one.
        results: The MXU results pushed and not yet popped, oldest first.
        transpose: The transpose staging array.
        bundles: Bundles issued so far.
        cycles: Cycles the issued bundles span, each taking one cycle plus its delay field, which counts the scheduled
            timing and leaves out any stall the hardware's own interlocks add.
        pending: Each DMA slot's unflushed descriptor as its bank, first row, row count and whether it stores, or None.
    """

    def __init__(self, dram: NDArray[np.uint8], *, arguments: tuple[int, ...]) -> None:
        self.dram = dram
        self.arguments = tuple(arguments) + (0,) * (4 - len(arguments))
        self.vmem = np.full((2, _VMEM_ROWS, _LANES), np.nan, dtype=np.float32)
        self.compute_bank = 0
        self.vregs = np.full((32, _SUBLANES, _LANES), np.nan, dtype=np.float32)
        self.sregs = [0, 0, 0, 0]
        self.weights: NDArray[np.float64] | None = None
        self.results: deque[NDArray[np.float64]] = deque()
        self.transpose = np.full((_LANES, _LANES), np.nan, dtype=np.float32)
        self.bundles = 0
        self.cycles = 0
        self.pending: list[tuple[int, int, int, bool] | None] = [None, None]

    def write_words(self, address: int, words: NDArray[np.uint16]) -> None:
        """Stages BF16 words in DRAM at a byte address."""
        data = np.asarray(words, dtype="<u2").view(np.uint8)
        self.dram[address:address + data.size] = data

    def read_words(self, address: int, count: int) -> NDArray[np.uint16]:
        """Returns `count` BF16 words of DRAM starting at a byte address."""
        return self.dram[address:address + 2 * count].view("<u2").copy()

    def run(self, words: list[int]) -> None:
        """Runs an image from its first bundle to its halt.

        Raises:
            EmulationError: If the image runs off its end, never halts, or does something the machine cannot run.
        """
        loops: list[dict[str, int]] = []
        pc = 0
        while True:
            if pc >= len(words):
                message = f"Unable to run bundle {pc}. The image must reach a halt, but ran off its end."
                raise EmulationError(message)
            self.bundles += 1
            if self.bundles > _STEP_LIMIT:
                message = f"Unable to reach a halt. The image must halt within {_STEP_LIMIT} bundles."
                raise EmulationError(message)
            word = words[pc]
            self.cycles += 1 + asm._delay_of(word)
            if word == asm.HALT_BUNDLE:
                if self.results:
                    message = f"Unable to halt with {len(self.results)} MXU results never popped."
                    raise EmulationError(message)
                return

            control = 0 if word >> 127 else _field(word, asm._F_SHIFT, asm.F_SLOT_W)
            control_op = control >> 24
            self._issue(word, pc, [frame["iv"] for frame in loops])
            if control_op == asm._OPCODES["lbegin"]:
                operand = control & 0xFFFFFF
                if len(loops) == 4:
                    message = f"Unable to open a loop at bundle {pc}. The loop stack holds 4."
                    raise EmulationError(message)
                loops.append({"start": pc + 1, "iv": (operand >> 20) & 0xF, "high": operand & 0xFFFF,
                              "step": (operand >> 16) & 0xF})
            elif control_op == asm._OPCODES["loop_end"]:
                frame = loops[-1]
                if frame["iv"] + frame["step"] < frame["high"]:
                    frame["iv"] += frame["step"]
                    pc = frame["start"]
                    continue
                loops.pop()
            pc += 1

    def _issue(self, word: int, pc: int, ivs: list[int]) -> None:
        """Runs one bundle, every slot reading the state before the bundle and every write landing after."""
        if word >> 127:
            self._descriptor(word, pc)
            return
        writes: list[tuple[int, NDArray[np.float32], NDArray[np.bool_] | None]] = []
        self._memory(word, pc, ivs, writes)
        self._matrix(word, pc, writes)
        self._vector(word, pc, writes)
        self._scalar(word, pc)
        control = _field(word, asm._F_SHIFT, asm.F_SLOT_W)
        if control >> 24 == asm._F_FLUSH:
            for slot in range(len(self.pending)):
                if (control >> slot) & 1:
                    self.pending[slot] = None
        if control >> 24 == asm._OPCODES["vmemswap"]:
            self.compute_bank ^= 1
        written = [register for register, _, _ in writes]
        if len(set(written)) != len(written):
            message = f"Unable to run bundle {pc}. It writes one vector register twice."
            raise EmulationError(message)
        for register, value, sublanes in writes:
            if sublanes is None:
                self.vregs[register] = to_bf16(value)
            else:
                self.vregs[register][sublanes] = to_bf16(value)[sublanes]

    def _descriptor(self, word: int, pc: int) -> None:
        """Moves one descriptor's rows between DRAM and the VMEM bank its address names."""
        descriptor = _field(word, asm._D_SHIFT, asm.D_WIRE_W)
        store = (descriptor >> 35) & 1
        address = (descriptor >> 20) & 0x3FFF
        rows = ((descriptor >> 4) & 0x7F) + 1
        base = self.sregs[(descriptor >> 2) & 0x3]
        stride = self.sregs[descriptor & 0x3]
        bank, row = address >> _BANK_SHIFT, address & (_VMEM_ROWS - 1)
        if row + rows > _VMEM_ROWS:
            message = f"Unable to run the descriptor at bundle {pc}. Its rows must stay inside one VMEM bank."
            raise EmulationError(message)
        for other, held in enumerate(self.pending):
            if held is not None and held[0] == bank and held[1] < row + rows and row < held[1] + held[2] \
                    and (store == 0 or not held[3]):
                message = (
                    f"Unable to run the descriptor at bundle {pc}. Its VMEM rows {row} through {row + rows - 1} race "
                    f"the unflushed descriptor on slot {other}, so a flush must cover that slot first."
                )
                raise EmulationError(message)
        self.pending[(word >> 125) & 1] = (bank, row, rows, bool(store))
        for index in range(rows):
            offset = base + index * stride
            if offset < 0 or offset + _ROW_BYTES > self.dram.size:
                message = f"Unable to run the descriptor at bundle {pc}. It reaches DRAM byte {offset}, outside DRAM."
                raise EmulationError(message)
            if store:
                words = (to_bf16(self.vmem[bank, row + index]).view(np.uint32) >> 16).astype("<u2")
                self.dram[offset:offset + _ROW_BYTES] = words.view(np.uint8)
            else:
                words = self.dram[offset:offset + _ROW_BYTES].view("<u2").astype(np.uint32)
                self.vmem[bank, row + index] = (words << 16).view(np.float32)

    def _memory(self, word: int, pc: int, ivs: list[int], writes: list) -> None:
        """Runs the X slot: a vld or vst of four consecutive rows in the compute bank, masked by sublane."""
        access = _field(word, asm._X_SHIFT, asm.X_SLOT_W)
        if not access:
            return
        store, vreg, row = (access >> 31) & 1, (access >> 26) & 0x1F, (access >> 13) & 0x1FFF
        agu, level, shift, mask = (access >> 12) & 1, (access >> 10) & 0x3, (access >> 7) & 0x7, access & 0xF
        if agu:
            if level >= len(ivs):
                message = f"Unable to run bundle {pc}. Its access walks loop level {level}, outside every loop."
                raise EmulationError(message)
            row += ivs[level] << shift
        sublanes = np.array([(mask >> sublane) & 1 == 1 for sublane in range(_SUBLANES)])
        rows = [(row + sublane) & (_VMEM_ROWS - 1) for sublane in range(_SUBLANES)]
        for slot, held in enumerate(self.pending):
            # A pending load races every access to its rows, and a pending store races only the writes.
            if held is not None and held[0] == self.compute_bank and (store or not held[3]) \
                    and any(held[1] <= target < held[1] + held[2] for target in rows):
                message = (
                    f"Unable to run bundle {pc}. Its {'vst' if store else 'vld'} of row {rows[0]} races the "
                    f"unflushed descriptor on slot {slot}, so a flush must cover that slot first."
                )
                raise EmulationError(message)
        bank = self.vmem[self.compute_bank]
        if store:
            for sublane, target in enumerate(rows):
                if sublanes[sublane]:
                    bank[target] = self.vregs[vreg][sublane]
        else:
            writes.append((vreg, bank[rows].copy(), sublanes))

    def _matrix(self, word: int, pc: int, writes: list) -> None:
        """Runs the M slot: loading a weight tile, pushing activations through it, or popping a result."""
        command = _field(word, asm._M_SHIFT, asm.M_SLOT_W)
        if not command:
            return
        op, register = command >> 5, command & 0x1F
        if op == asm._OPCODES["vmatload"]:
            if register > 28:
                message = f"Unable to load weights from v{register} at bundle {pc}. A tile's four registers must fit."
                raise EmulationError(message)
            self.weights = self.vregs[register:register + 4].reshape(4 * _SUBLANES, _LANES).astype(np.float64)
        elif op == asm._OPCODES["vmatpush"]:
            if self.weights is None:
                message = f"Unable to push at bundle {pc}. A vmatpush must follow a vmatload."
                raise EmulationError(message)
            self.results.append(self.vregs[register].astype(np.float64) @ self.weights)
            if len(self.results) * _SUBLANES > _MXU_FIFO_ENTRIES:
                message = f"Unable to push at bundle {pc}. The MXU output FIFO overflows, dropping results."
                raise EmulationError(message)
        elif op == asm._OPCODES["vmatpop"]:
            if not self.results:
                message = f"Unable to pop at bundle {pc}. A vmatpop must follow a vmatpush."
                raise EmulationError(message)
            writes.append((register, self.results.popleft().astype(np.float32), None))

    def _vector(self, word: int, pc: int, writes: list) -> None:
        """Runs the E slot's arithmetic, reductions and transposes."""
        operation = _field(word, asm._E_SHIFT, asm.E_SLOT_W)
        if not operation:
            return
        op, rd, ra, rb = operation >> 15, (operation >> 10) & 0x1F, (operation >> 5) & 0x1F, operation & 0x1F
        a = self.vregs[ra].astype(np.float64)
        b = self.vregs[rb].astype(np.float64)
        codes = asm._OPCODES
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if op == codes["vtxin"]:
                self.transpose[4 * rb:4 * rb + 4] = self.vregs[ra]
                return
            results = {
                codes["vadd"]: lambda: a + b,
                codes["vsub"]: lambda: a - b,
                codes["vmul"]: lambda: a * b,
                codes["vmax"]: lambda: np.maximum(a, b),
                codes["vmin"]: lambda: np.minimum(a, b),
                codes["vmov"]: lambda: a,
                codes["vgelu"]: lambda: 0.5 * a * (1.0 + _ERF(a / math.sqrt(2.0))),
                codes["vexp"]: lambda: np.where(a > 0, 1.0, np.exp(a)),
                codes["vrecip"]: lambda: 1.0 / a,
                codes["vrsqrt"]: lambda: np.where(a > 0, 1.0 / np.sqrt(np.where(a > 0, a, 1.0)), np.nan),
                codes["vredsum"]: lambda: np.full_like(a, a.sum()),
                codes["vredmax"]: lambda: np.full_like(a, a.max()),
                codes["vtxout"]: lambda: self.transpose[:, 4 * rb:4 * rb + 4].T.astype(np.float64),
            }
            if op not in results:
                message = f"Unable to run bundle {pc}. E opcode {op} names no operation."
                raise EmulationError(message)
            writes.append((rd, results[op]().astype(np.float32), None))

    def _scalar(self, word: int, pc: int) -> None:
        """Runs the S slot. The scheduler spaces every scalar reader past the write it reads, so writes land at once."""
        operation = _field(word, asm._S_SHIFT, asm.S_SLOT_W)
        if not operation or not (operation >> 26) & 1:
            return
        op, rd, rs, immediate = (operation >> 23) & 0x7, (operation >> 21) & 0x3, (operation >> 19) & 0x3, operation & 0xFFFF
        codes = asm._OPCODES
        if op == codes["smovi"]:
            self.sregs[rd] = immediate
        elif op == codes["smov_arg"]:
            self.sregs[rd] = self.arguments[rs]
        elif op == codes["saddi"]:
            signed = immediate - (1 << 16) if immediate >> 15 else immediate
            self.sregs[rd] = (self.sregs[rs] + signed) & 0xFFFF_FFFF
        else:
            message = f"Unable to run bundle {pc}. Scalar opcode {op} is not modelled."
            raise EmulationError(message)


def read_image(path: pathlib.Path) -> list[int]:
    """Returns an image's bundle words, each sixteen little-endian bytes."""
    data = path.read_bytes()
    return [int.from_bytes(data[offset:offset + 16], "little") for offset in range(0, len(data), 16)]
