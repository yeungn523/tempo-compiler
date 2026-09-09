"""Provides a kernel surface where layouts are written and addresses are derived.

The programmer writes which layout a tensor is in, which VMEM stride it is staged at, where a relayout happens and
how the launch is tiled. A layout mismatch or a non-affine index is refused rather than repaired, because a
repair is a launch.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

from bf16 import _NUM_SUBLANES, _ROW_BYTES
from layout import LayoutError, Tiled, VmemAllocator, VmemRegion
import asm as asm_module
from runtime import minitpu


ops = asm_module.AsmBuilder()


@dataclass(frozen=True)
class Vmem:
    """A VMEM staging layout.

    There is one, and it holds no choice any more. A staged tensor lands each
    token's feature tiles CONSECUTIVELY, so the reduction that walks them reads
    at stride one.
    """
@dataclass
class Ref:
    """A staged tensor: DMEM layout, VMEM region, and how to reach a row of it."""

    name: str
    dmem: Tiled
    region: VmemRegion
    kernel: "Kernel"

    def row(self, level: int) -> "Access":
        """One VREG's worth, with the four sublanes walking feature tiles.

        A token owns `region.tile_stride` consecutive rows, so the induction variable
        reaches token t with a SHIFT, and the four sublanes then read four
        adjacent tiles of it.
        """
        return Access(self, level)

    @property
    def shift(self) -> int:
        """The left shift the AGU applies to the induction variable.

        The region's stride is a power of two precisely so this exists: the AGU
        shifts the index, it does not multiply by it.
        """
        return (self.region.tile_stride - 1).bit_length()


@dataclass(frozen=True)
class Access:
    """A derived VREG access. Nothing here was written by hand."""

    ref: Ref
    level: int

    def base(self, group: int) -> int:
        """VMEM row for sublane group `group` of this access.

        A VREG spans `_NUM_SUBLANES` adjacent feature tiles, so the next one
        starts that many rows on -- no stride enters, which is the whole change.
        """
        return self.ref.region.row(0) + group * _NUM_SUBLANES


@dataclass
class Kernel:
    """A program under construction, and the state a derivation needs."""

    name: str
    vmem: VmemAllocator = field(default_factory=VmemAllocator)
    bundles: dict = field(default_factory=dict)
    pc: int = 0
    level: int = -1
    _args: dict = field(default_factory=dict)
    _stride_sreg: int | None = None

    # -- what the programmer writes ----------------------------------------

    def arg(self, name: str, dmem: Tiled, *, slot: int, sreg: int) -> Tiled:
        """Binds a launch argument to a scalar register."""
        self._args[name] = (slot, sreg, dmem)
        return dmem

    @contextmanager
    def loop(self, name: str, trips: int):
        """A loop level. Nesting order IS `agu_level`; level 0 is outermost."""
        self.level += 1
        if self.level >= (1 << asm_module.LEVEL_SEL_W):
            message = (
                f"Unable to open loop {name!r} at level {self.level}. `agu_level` "
                f"is {asm_module.LEVEL_SEL_W} bits, so the loop stack is "
                f"{1 << asm_module.LEVEL_SEL_W} deep."
            )
            raise LayoutError(message)
        self._emit(f=ops.lbegin(trips))
        try:
            yield self.level
        finally:
            self._emit(f=ops.loop_end())
            self.level -= 1

    def stage(self, name: str, dmem: Tiled, *, tiles: int, rows: int,
              vmem: Vmem, sreg: int) -> Ref:
        """Brings `rows` tokens of `tiles` feature tiles into VMEM. THIS is the relayout.

        One descriptor per TOKEN, gathering that token's tiles -- which are
        `rows` apart in DMEM -- into consecutive VMEM rows. A descriptor names
        both its DMEM stride and its destination row, so the relayout costs
        nothing but the choice to make it.
        """
        region = self.vmem.token_major_region(name, rows, tiles=tiles)
        for token in range(rows):
            self._emit(d=ops.dma_load(token % 2, region.row(token), tiles,
                                      sreg, self._stride()))
            self._emit(s=ops.saddi(sreg, sreg, _ROW_BYTES), advance=2)
            if token % 2 or token == rows - 1:
                self._emit(f=ops.flush_slot(3))
        # The descriptors walked one row per token; a head is `tiles` times that.
        # The remainder is what puts the pointer on the next head.
        self._emit(s=ops.saddi(sreg, sreg, (tiles - 1) * rows * _ROW_BYTES),
                   advance=2)
        return Ref(name, dmem, region, self)

    def drain(self, ref: Ref, *, tiles: int, rows: int, sreg: int) -> None:
        """Writes a staged region back out, one descriptor per token."""
        for token in range(rows):
            self._emit(d=ops.dma_store(token % 2, ref.region.row(token), tiles,
                                       sreg, self._stride()))
            self._emit(s=ops.saddi(sreg, sreg, _ROW_BYTES), advance=2)
            if token % 2 or token == rows - 1:
                self._emit(f=ops.flush_slot(3))
        # As in `stage`: the pointer is walked on to the next head.
        self._emit(s=ops.saddi(sreg, sreg, (tiles - 1) * rows * _ROW_BYTES),
                   advance=2)

    def load(self, access: Access, vregs: list[int]) -> None:
        """Derives one `vld` per VREG. Base, shift and level are ours."""
        for group, vreg in enumerate(vregs):
            self._emit(x=ops.vld(vreg, access.base(group), agu=True,
                                 shift=access.ref.shift, level=access.level))

    def store(self, access: Access, vregs: list[int]) -> None:
        """Derives one `vst` per VREG, the same way."""
        for group, vreg in enumerate(vregs):
            self._emit(x=ops.vst(vreg, access.base(group), agu=True,
                                 shift=access.ref.shift, level=access.level))

    def compute(self, word: int) -> None:
        """One E-slot op, placed at the next bundle."""
        self._emit(e=word)

    # -- what the compiler does --------------------------------------------

    def _stride(self) -> int:
        """The scalar register holding a descriptor's DMEM row stride."""
        if self._stride_sreg is None:
            raise LayoutError("prologue() must run before any descriptor")
        return self._stride_sreg

    def prologue(self, *, stride_sreg: int = 2, stride_rows: int = 1) -> None:
        """Emits the DMEM row stride and binds every declared argument.

        Args:
            stride_sreg: Which scalar register holds the stride.
            stride_rows: DMEM rows between the rows one descriptor gathers. A
                token's feature tiles are `tokens` rows apart, so this is the
                token count, and it is why the stride is a register rather than
                a fixed row stride.
        """
        self._stride_sreg = stride_sreg
        self._emit(s=ops.smovi(stride_sreg, stride_rows * _ROW_BYTES))
        for _, (slot, sreg, _) in self._args.items():
            self._emit(s=ops.smov_arg(sreg, slot))
        self.pc += 1                       # scalar writeback before first use

    def _emit(self, *, advance: int = 1, **slots) -> None:
        self.bundles[self.pc] = slots
        self.pc += advance

    def gap(self, cycles: int) -> None:
        """Explicit slack. The loop back edge needs it; `schedule()` cannot see it."""
        self.pc += cycles

    def assemble(self):
        """Hands the emitted bundles to the assembler."""
        self._emit(f=ops.halt())
        emitted, count = dict(self.bundles), self.pc

        def program() -> None:
            for index in range(count):
                minitpu.bundle(**emitted.get(index, {}))
        program.__name__ = self.name
        return minitpu.asm(program)
