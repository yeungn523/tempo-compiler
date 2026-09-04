"""A kernel surface where layouts are written and addresses are derived.

Everything here follows from one division, which the measurements in
SESSION_CHECKLIST.md argue for rather than assume:

    the programmer writes    which layout a tensor is in, which VMEM pitch it is
                             staged at, where a relayout happens, and how the
                             launch is tiled
    the compiler derives     every descriptor field, every vld and vst field, the
                             loop level an AGU walk binds to, and what the access
                             costs in bank cycles
    the compiler refuses     a layout mismatch and a non-affine index; it never
                             repairs either, because a repair is a launch
    nobody decides           which layout is best -- there is no cost model here,
                             and Mosaic does not have one either

The surface is small because the machine is. A VMEM row IS one lane group and rows
are the sublane axis, so the only freedom in a VREG access is which rows the four
sublanes take: one stride, plus which loop level the base walks. That is why this
file has `Vmem(pitch=...)` and `Ref.row(iv)` and nothing resembling Mosaic's
`bitwidth`, `tiling` or `implicit_dim`.

`tools/check_dsl.py` builds the packed softmax through this and holds the result
against `kernels.build_softmax_packed`, bundle for bundle.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

from bf16 import _NUM_SUBLANES, _ROW_BYTES
from layout import LayoutError, Tiled, VmemAllocator, VmemRegion
import asm as asm_module
from runtime import minitpu

#: `minitpu` above is the live runtime namespace, and `asm(program)` REPLACES it
#: in this module's globals for the duration of a capture -- which is how
#: `program` below reaches the builder. Importing it rather than relying on that
#: injection keeps the name defined, which `tools/check_names.py` requires and
#: which is not pedantry: a global that only exists during a capture is exactly
#: the kind of name that goes stale without anything noticing.

#: Words are built through a private assembler instance rather than the `minitpu`
#: namespace, which only becomes an `AsmBuilder` inside `asm(function)`. The two
#: produce identical words -- `AsmBuilder`'s methods are pure encoders -- and this
#: way a kernel can be constructed before anything is being captured.
ops = asm_module.AsmBuilder()


@dataclass(frozen=True)
class Vmem:
    """A VMEM staging layout. The pitch is the programmer's, deliberately.

    `pitch=None` asks for the smallest conflict-free choice. Passing a colliding
    pitch is legal and costs `gcd(pitch, NUM_SUBLANES)` cycles per access -- the
    compiler prices it and emits it, and `tools/demo_bank_conflicts.py` measures
    3.13x on a board for the worst case. A surface that silently substituted the
    good pitch would make the bad one unobservable.
    """

    pitch: int | None = None


@dataclass
class Ref:
    """A staged tensor: DMEM layout, VMEM region, and how to reach a row of it."""

    name: str
    dmem: Tiled
    region: VmemRegion
    kernel: "Kernel"

    def row(self, level: int) -> "Access":
        """One VREG's worth, with the four sublanes walking feature tiles.

        The base walks `level`'s induction variable by one row per step, because
        one token is one row of every tile's base. The sublanes then reach four
        consecutive tiles at the region's pitch.
        """
        return Access(self, level)

    @property
    def conflict(self) -> int:
        """Cycles one such access costs."""
        return math.gcd(self.region.pitch, _NUM_SUBLANES)


@dataclass(frozen=True)
class Access:
    """A derived VREG access. Nothing here was written by hand."""

    ref: Ref
    level: int

    def base(self, group: int) -> int:
        """VMEM row for sublane group `group` of this access."""
        return self.ref.region.row(0) + group * _NUM_SUBLANES * self.ref.region.pitch


@dataclass
class Kernel:
    """A program under construction, and the state a derivation needs."""

    name: str
    vmem: VmemAllocator = field(default_factory=VmemAllocator)
    bundles: dict = field(default_factory=dict)
    pc: int = 0
    level: int = -1
    _args: dict = field(default_factory=dict)
    _pitch_sreg: int | None = None

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
        """Brings `tiles` feature tiles into VMEM. THIS is the relayout.

        One descriptor per tile, each landing at its own pitch, which is the whole
        of a re-pitch: a descriptor names its own destination row, so changing the
        VMEM layout on the way in costs nothing but the choice.
        """
        conflict_free = vmem.pitch is None
        region = self.vmem.region(name, rows, tiles=tiles,
                                  conflict_free=conflict_free)
        if vmem.pitch is not None and vmem.pitch != region.pitch:
            region = VmemRegion(name, region.base, rows, vmem.pitch, tiles)
        for tile in range(tiles):
            self._emit(d=ops.dma_load(tile % 2, region.row(tile), rows,
                                          sreg, self._pitch()))
            self._emit(s=ops.saddi(sreg, sreg, rows * _ROW_BYTES), advance=2)
            if tile % 2 or tile == tiles - 1:
                self._emit(f=ops.flush_slot(3))
        return Ref(name, dmem, region, self)

    def drain(self, ref: Ref, *, tiles: int, rows: int, sreg: int) -> None:
        """Writes a staged region back out, one descriptor per tile."""
        for tile in range(tiles):
            self._emit(d=ops.dma_store(tile % 2, ref.region.row(tile), rows,
                                           sreg, self._pitch()))
            self._emit(s=ops.saddi(sreg, sreg, rows * _ROW_BYTES), advance=2)
            if tile % 2 or tile == tiles - 1:
                self._emit(f=ops.flush_slot(3))

    def load(self, access: Access, vregs: list[int]) -> None:
        """Derives one `vld` per VREG. Base, stride, level and spacing are ours."""
        for group, vreg in enumerate(vregs):
            self._emit(x=ops.vld(vreg, access.base(group), agu=True, shift=0,
                                     level=access.level,
                                     stride=access.ref.region.pitch),
                       advance=access.ref.conflict)

    def store(self, access: Access, vregs: list[int]) -> None:
        """Derives one `vst` per VREG, the same way."""
        for group, vreg in enumerate(vregs):
            self._emit(x=ops.vst(vreg, access.base(group), agu=True, shift=0,
                                     level=access.level,
                                     stride=access.ref.region.pitch),
                       advance=access.ref.conflict)

    def compute(self, word: int) -> None:
        """One E-slot op, placed at the next bundle."""
        self._emit(e=word)

    # -- what the compiler does --------------------------------------------

    def _pitch(self) -> int:
        """The scalar register holding one DMEM row's byte pitch."""
        if self._pitch_sreg is None:
            raise LayoutError("prologue() must run before any descriptor")
        return self._pitch_sreg

    def prologue(self, *, pitch_sreg: int = 3) -> None:
        """Emits the row pitch and binds every declared argument."""
        self._pitch_sreg = pitch_sreg
        self._emit(s=ops.smovi(pitch_sreg, _ROW_BYTES))
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
