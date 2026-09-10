"""Derives the tiled GEMM's bundle placement from the published ISA contract and the loop nest it walks.

Every placement constant that follows from the machine is computed from the writeback offsets and
controller occupancies `compiler/minitpuisa.td` publishes through `board_package/isa_model.py`, and the two
that remain written out are scheduling policy their docstrings name as such. Where each operand is
re-staged, and how many output tiles the accumulators must hold, follow from the axes each address walks
against the loops the nest carries, which is what lets a nesting order reach the image. Correctness is
judged by reproducing `board_package/kernels.py`'s bundle stream exactly, because that stream is the only
one a board is known to accept.
"""

from __future__ import annotations

import pathlib
import sys

# Places this package on sys.path, which `asm`, `bf16` and `isa_model` are imported from below and which is not
# installed, so this must precede those imports rather than sit beneath them.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import asm as asm_module  # noqa: E402
from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _NUM_SUBLANES,
    _ROW_BYTES,
    _STEPS,
    _TILE,
    _TILE_ROWS,
    _TILE_VREGS,
    _TOKEN_GROUPS,
    _VMEM_SLOT_STRIDE,
    _ZERO_TILE_BYTES,
)
from isa_model import MATRIX_OCCUPANCY, RTL_PARAMS  # noqa: E402
from layout import RegisterMap, VmemAllocator, register_map  # noqa: E402

_REGISTERS: RegisterMap = register_map(block_m=_BLOCK_TOKENS, block_n=_TILE)
"""Where a block's operands live in the register file, which `layout.register_map` derives from its shape.

A block is the one shape the emitter places, so the map is asked once here rather than per step. The accumulators
it leaves room for bound what a nest may keep resident, since this emitter has no path that accumulates in VMEM.
"""

_MATLOAD_AT: int = (_TILE_VREGS - 1) + RTL_PARAMS["WB_W_VLD"] + 1
"""Offset the weight load issues at, being the first cycle every weight register has landed.

The weight loads occupy the step's first `_TILE_VREGS` offsets, and a load's earliest dependent consumer sits one
cycle past its writeback reservation.
"""

_PUSH_AT: int = _MATLOAD_AT + MATRIX_OCCUPANCY["vmatload"] + 1
"""Offset the first activation push issues at, being the first cycle the matrix controller is free again.

A command issued at one cycle holds the controller for its occupancy, so the next becomes legal one cycle later.
"""

_PUSH_II: int = MATRIX_OCCUPANCY["vmatpush"] + 1
"""Offsets between consecutive pushes, being the interval the controller's occupancy allows."""

_POP_II: int = 5
"""Offsets between consecutive pops, which is scheduling policy rather than a consequence of the ISA.

A pop holds the controller for seven cycles, so the ISA alone would place the next one eight offsets later. Placing
them closer states an intent to issue and leaves the real distance to `asm.schedule`, which prices it from that same
occupancy.
"""

_ACCUMULATE_POP_GAP: int = 2
"""Offsets left after a group's accumulate before the next pop, which is scheduling policy.

The adds hold the result registers the following pop overwrites, so the gap keeps them clear of its write window.
"""

_ZERO_ROWS: int = _TILE
"""Rows the clearing tile occupies, which one descriptor fills before any column tile runs."""


_OUTPUT_ROWS: int = _TOKEN_GROUPS * _TILE
"""Rows the resident output tiles occupy, which one descriptor drains."""

_DRAIN_POINTER: int = 1
"""Scalar register holding the drain buffer's host address."""

_STRIDE_REGISTER: int = 3
"""Scalar register holding one DMEM row's byte stride, which every descriptor reads."""

_SCALAR_GAP: int = asm_module.W_SCALAR + 1
"""Bundles a scalar write must clear before a descriptor may read the register, the unit interlocking nothing."""

_REDUCTION_AXIS: str = "k"
"""Axis the contraction walks."""

_COLUMN_AXIS: str = "n"
"""Axis the output columns walk."""


@dataclass(frozen=True)
class LoopLevel:
    """Describes one loop of the emitted nest.

    Attributes:
        axis: Problem axis the loop walks, being "n" for output columns and "k" for the reduction. The token axis
            is absent because the host issues one launch per token block rather than the image walking them.
        trips: Iterations the loop runs.
    """

    axis: str
    trips: int


@dataclass(frozen=True)
class StagedOperand:
    """Describes one tensor staged into VMEM, and how its descriptor reaches it.

    Attributes:
        name: What the operand is, for the messages a staging decision appears in.
        axes: Problem axes the operand's address depends on, which is what decides where it is re-staged.
        argument: Launch argument holding the operand's base, which a loop it is invariant to rebinds from.
        dma_slot: Descriptor slot the transfer is issued on.
        vmem_rows: First VMEM row of the operand's tile, one per step of a block.
        rows: Rows one descriptor moves.
        pointer: Scalar register holding the operand's host address.
    """

    name: str
    axes: frozenset[str]
    argument: int
    dma_slot: int
    vmem_rows: tuple[int, ...]
    rows: int
    pointer: int

    @property
    def advance(self) -> int:
        """Returns the bytes the pointer moves once its descriptor has read it."""
        return self.rows * _ROW_BYTES


def vmem_map(*, output_tiles: int = 1) -> dict[str, tuple[int, ...]]:
    """Returns the first VMEM row of every region a launch stages, laid out end to end from row zero.

    Args:
        output_tiles: Output tiles one block covers, being the weight tiles a step stages and the tiles it drains.

    Returns:
        The clearing tile's row, each step's activation and weight rows, and the output's row.
    """
    vmem = VmemAllocator()
    rows = {"zero": (vmem.region("zero", _ZERO_ROWS).base,)}
    activation, weight = [], []
    for step in range(_STEPS):
        activation.append(vmem.region(f"activation {step}", _TOKEN_GROUPS * _TILE).base)
        weight.append(vmem.region(f"weight {step}", _TILE * output_tiles).base)
    rows["activation"] = tuple(activation)
    rows["weight"] = tuple(weight)
    rows["output"] = (vmem.region("output", _TOKEN_GROUPS * _TILE * output_tiles).base,)
    vmem.check_disjoint()
    return rows


def gemm_operands(*, rows: dict[str, tuple[int, ...]]) -> tuple[StagedOperand, ...]:
    """Returns every operand a launch stages, in the order their descriptors issue.

    Args:
        rows: Where each region sits, as `vmem_map` laid it out.

    Returns:
        The activation and the weight.
    """
    activation = StagedOperand(
        name="activation", axes=frozenset({_REDUCTION_AXIS}), argument=0, dma_slot=0,
        vmem_rows=rows["activation"], rows=_TOKEN_GROUPS * _TILE, pointer=0)
    weight = StagedOperand(
        name="weight", axes=frozenset({_REDUCTION_AXIS, _COLUMN_AXIS}), argument=1, dma_slot=1,
        vmem_rows=rows["weight"], rows=_TILE, pointer=2)
    return activation, weight


def staging_levels(*, nest: tuple[LoopLevel, ...],
                   operands: tuple[StagedOperand, ...]) -> dict[int, tuple[StagedOperand, ...]]:
    """Returns the operands re-staged at each loop level, keyed by that level's depth in the nest.

    Notes:
        An operand is re-staged at the innermost loop whose axis its address depends on, because every loop inside
        that one leaves the address unchanged and would re-read what VMEM already holds. Which loops those are is
        what the nesting order decides, so this is where a loop order becomes a different image: a reduction inside
        the column loop leaves the left operand re-read once per column tile, and a column loop inside the reduction
        stages it once and reuses it.

    Args:
        nest: The emitted loops, outermost first.
        operands: The operands to place.

    Returns:
        The operands staged at each level, keyed by depth.

    Raises:
        ValueError: If an operand's address depends on no loop in the nest, which leaves it nowhere to be staged.
    """
    levels: dict[int, list[StagedOperand]] = {}
    for operand in operands:
        walking = [depth for depth, loop in enumerate(nest) if loop.axis in operand.axes]
        if not walking:
            message = (
                f"Unable to stage the {operand.name} operand, whose address walks "
                f"{', '.join(sorted(operand.axes))} while the nest walks "
                f"{', '.join(loop.axis for loop in nest)}. An operand no loop walks is staged once before the "
                f"nest, which this emitter does not place."
            )
            raise ValueError(message)
        levels.setdefault(max(walking), []).append(operand)
    return {depth: tuple(staged) for depth, staged in sorted(levels.items())}


def resident_tiles(*, nest: tuple[LoopLevel, ...]) -> int:
    """Returns the output tiles the accumulators must hold at once for one pass of the nest.

    Notes:
        An output tile is invariant to the reduction, so it stays live across every reduction loop enclosing it.
        A column loop nested inside one therefore keeps all of its tiles live at once, while a column loop outside
        every reduction loop keeps one. This is the cost a nesting order charges against the register file, and it
        is what makes walking the columns inside the contraction expensive rather than free.

    Args:
        nest: The emitted loops, outermost first.

    Returns:
        The tiles held at once, being one when no column loop sits inside a reduction loop.
    """
    outermost_reduction = next((depth for depth, loop in enumerate(nest)
                                if loop.axis == _REDUCTION_AXIS), len(nest))
    tiles = 1
    for loop in nest[outermost_reduction:]:
        if loop.axis == _COLUMN_AXIS:
            tiles *= loop.trips
    return tiles


def emit_gemm_step(*, activation_row: int, weight_row: int, close_loop: bool) -> dict[int, dict[str, int]]:
    """Returns one accumulation step's bundles, keyed by their offset from the step's first bundle.

    Notes:
        One rule places the matrix commands. Push as early as the output FIFO allows and pop only when a further
        push would overflow it, because a pop stalls until the array produces the result it takes while a push is
        what the array waits on. The FIFO's depth is read rather than assumed, so a deeper one changes the
        placement without changing this function.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.

    Returns:
        The step's bundles, keyed by offset, each holding the slot words that bundle issues.
    """
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so a push and a load co-issue where both fit."""
        bundles.setdefault(offset, {}).update(slots)

    for vreg in range(_TILE_VREGS):
        put(offset=vreg,
            x=builder.vld(vreg=_REGISTERS.weight + vreg, row=weight_row + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(offset=_TILE_VREGS + vreg,
            x=builder.vld(vreg=_REGISTERS.activation + vreg, row=activation_row + _NUM_SUBLANES * vreg))
    put(offset=_MATLOAD_AT, m=builder.vmatload(vreg=_REGISTERS.weight))

    def stage_next(*, group: int, vreg: int) -> int | None:
        """Returns the load of the next token group's tile, or None once the last group is staged."""
        if group + 1 >= _TOKEN_GROUPS:
            return None
        return builder.vld(vreg=_REGISTERS.activation + _TILE_VREGS * (group + 1) + vreg,
                           row=activation_row + _TILE_ROWS * (group + 1) + _NUM_SUBLANES * vreg)

    _place_matrix_commands(put=put, builder=builder, stage_next=stage_next, close_loop=close_loop)
    return bundles


def step_stream(*, activation_row: int, weight_row: int, close_loop: bool) -> tuple[int, ...]:
    """Returns one accumulation step as the contiguous bundle words it places, zero filling the offsets it leaves.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.

    Returns:
        The bundle words, in placement order.
    """
    return _flatten(emit_gemm_step(activation_row=activation_row, weight_row=weight_row, close_loop=close_loop))


def emit_looped_gemm(*, column_tiles: int, groups: int, loop_blocks: int,
                     fuse_groups: bool = False) -> dict[int, dict[str, int]]:
    """Returns the tiled GEMM's bundles, keyed by their offset from the image's first bundle.

    Notes:
        Staging is overlapped onto compute by double buffering. A block computes out of one VMEM slot while the
        next block's operands land in the other, and `vmemswap` exchanges them once per block.

    Args:
        column_tiles: Output column tiles the product covers, being the outer loop's trip count.
        groups: Reduction groups each column tile accumulates, being the middle loop's trip count.
        loop_blocks: Accumulation blocks one group carries, which the two way unroll halves into a trip count.
        fuse_groups: Determines whether the accumulators stay resident across every group and drain once per
            column tile, summing the contraction on the device in BF16, rather than draining a partial per
            group for the host to sum in float32.

    Returns:
        The image's bundles, keyed by offset, each holding the slot words that bundle issues.

    Raises:
        ValueError: If a group carries an odd number of blocks, which the two way unroll cannot halve, or if the
            nest holds more output tiles resident than the register map carries, or if it stages an operand
            anywhere but the innermost loop.
    """
    if loop_blocks % 2:
        message = (
            f"Unable to unroll a group of {loop_blocks} blocks two ways. Double buffering alternates the VMEM "
            f"slot every block and a descriptor names its slot outright, so a group must carry an even count."
        )
        raise ValueError(message)

    nest = (LoopLevel(axis=_COLUMN_AXIS, trips=column_tiles),
            LoopLevel(axis=_REDUCTION_AXIS, trips=groups),
            LoopLevel(axis=_REDUCTION_AXIS, trips=loop_blocks // 2))
    rows = vmem_map()
    staged = _innermost_staging(nest=nest, operands=gemm_operands(rows=rows))

    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        bundles.setdefault(offset, {}).update(slots)

    at = _emit_prologue(put=put, builder=builder, at=0, rows=rows, weight=staged[-1])

    put(offset=at, f=builder.lbegin(nest[0].trips))
    at += 1
    at = _emit_rewind(put=put, builder=builder, at=at, loop=nest[0], operands=staged)

    # Clears the accumulators once per column tile, so they carry every group and the contraction sums in BF16 on
    # the device.
    if fuse_groups:
        at = _emit_clear(put=put, builder=builder, at=at, rows=rows)

    put(offset=at, f=builder.lbegin(nest[1].trips))
    at += 1
    # Steps back over the block the column loop's head advanced past, so each group re-reads its first block.
    for operand in staged:
        put(offset=at, s=builder.saddi(rd=operand.pointer, rs=operand.pointer,
                                       value=-operand.advance * _STEPS))
        at += 1
    at += _SCALAR_GAP - 1
    for step in range(_STEPS):
        at = _emit_stage(put=put, builder=builder, at=at, step=step, vmem_slot=0, operands=staged)
        if step + 1 < _STEPS:
            put(offset=at, f=builder.flush_slot(mask=3))
            at += 1
    if not fuse_groups:
        at = _emit_clear(put=put, builder=builder, at=at, rows=rows)

    put(offset=at, f=builder.lbegin(nest[2].trips))
    at += 1
    for compute_slot in range(2):
        at = _emit_block(put=put, builder=builder, at=at, compute_slot=compute_slot, operands=staged,
                         rows=rows)
    put(offset=at, f=builder.loop_end())
    at += 1

    # Returns both descriptor slots, which the group's trailing prefetch still owns, for the drain to use.
    put(offset=at, f=builder.flush_slot(mask=3))
    at += 1
    if not fuse_groups:
        at = _emit_drain(put=put, builder=builder, at=at, rows=rows)
    put(offset=at, f=builder.loop_end())
    at += 1
    if fuse_groups:
        at = _emit_drain(put=put, builder=builder, at=at, rows=rows)
    put(offset=at, f=builder.loop_end())
    at += 1
    put(offset=at, f=builder.halt())
    return bundles


def looped_gemm_stream(*, column_tiles: int, groups: int, loop_blocks: int,
                       fuse_groups: bool = False) -> tuple[int, ...]:
    """Returns the tiled GEMM as the contiguous bundle words it places, zero filling the offsets it leaves.

    Args:
        column_tiles: Output column tiles the product covers.
        groups: Reduction groups each column tile accumulates.
        loop_blocks: Accumulation blocks one group carries.
        fuse_groups: Determines whether the accumulators drain once per column tile rather than once per group.

    Returns:
        The bundle words, in placement order.
    """
    return _flatten(emit_looped_gemm(column_tiles=column_tiles, groups=groups, loop_blocks=loop_blocks,
                                     fuse_groups=fuse_groups))


def _innermost_staging(*, nest: tuple[LoopLevel, ...],
                       operands: tuple[StagedOperand, ...]) -> tuple[StagedOperand, ...]:
    """Returns the operands staged at the innermost loop, refusing a nest this emitter cannot place.

    Args:
        nest: The emitted loops, outermost first.
        operands: The operands to place.

    Returns:
        The operands the innermost loop re-stages.

    Raises:
        ValueError: If the nest holds more output tiles resident than the register map carries, or if it assigns
            an operand to any loop but the innermost one.
    """
    tiles = resident_tiles(nest=nest)
    if tiles > _REGISTERS.resident:
        message = (
            f"Unable to walk a nest holding {tiles} output tiles resident. The register map carries "
            f"{_REGISTERS.resident}, so a column loop inside the contraction needs an allocator to widen it. "
            f"Walk the columns outside the contraction, which holds one tile however many the product covers."
        )
        raise ValueError(message)

    levels = staging_levels(nest=nest, operands=operands)
    hoisted = {depth: staged for depth, staged in levels.items() if depth != len(nest) - 1}
    if hoisted:
        placed = ", ".join(f"{operand.name} at depth {depth}"
                           for depth, staged in sorted(hoisted.items()) for operand in staged)
        message = (
            f"Unable to stage {placed}. This emitter issues every descriptor from the innermost loop, so an "
            f"operand the nest leaves invariant there has nowhere to be placed."
        )
        raise ValueError(message)
    return levels[len(nest) - 1]


def _emit_prologue(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                   rows: dict[str, tuple[int, ...]], weight: StagedOperand) -> int:
    """Binds the launch arguments and fills the clearing tile, returning the next free offset."""
    put(offset=at, s=builder.smovi(rd=_STRIDE_REGISTER, value=_ROW_BYTES))
    at += 1
    put(offset=at, s=builder.smov_arg(rd=_DRAIN_POINTER, arg=2))
    at += _SCALAR_GAP
    put(offset=at, d=builder.dma_load(slot=0, vmem_address=rows["zero"][0], rows=_ZERO_ROWS,
                                      base_sreg=_DRAIN_POINTER, stride_sreg=_STRIDE_REGISTER))
    at += 1
    put(offset=at, f=builder.flush_slot(mask=1))
    at += 1
    put(offset=at, s=builder.smov_arg(rd=weight.pointer, arg=1))
    at += 1
    # Steps the drain pointer past the clearing tile, which is read once and never written.
    put(offset=at, s=builder.saddi(rd=_DRAIN_POINTER, rs=_DRAIN_POINTER, value=_ZERO_TILE_BYTES))
    at += _SCALAR_GAP
    put(offset=at, s=builder.saddi(rd=weight.pointer, rs=weight.pointer, value=_BLOCK_WEIGHT_BYTES))
    return at + _SCALAR_GAP


def _emit_rewind(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, loop: LoopLevel,
                 operands: tuple[StagedOperand, ...]) -> int:
    """Rebinds every operand this loop is invariant to, and primes it for the prefetch, returning the next offset.

    Notes:
        An operand whose address does not walk this loop's axis is read again from its base on every iteration,
        so the pointer returns to the launch argument rather than walking on. One whose address does walk it is
        left alone, because the next iteration wants exactly where the last one stopped. The rebind is followed by
        one block's advance, which is the lead the innermost loop's prefetch runs at.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        at: Offset the rewind starts at.
        loop: The loop whose head this is.
        operands: The operands the innermost loop stages.

    Returns:
        The next free offset.
    """
    for operand in operands:
        if loop.axis in operand.axes:
            continue
        put(offset=at, s=builder.smov_arg(rd=operand.pointer, arg=operand.argument))
        at += _SCALAR_GAP
        put(offset=at, s=builder.saddi(rd=operand.pointer, rs=operand.pointer,
                                       value=operand.advance * _STEPS))
        at += _SCALAR_GAP
    return at


def _emit_stage(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, step: int,
                vmem_slot: int, operands: tuple[StagedOperand, ...]) -> int:
    """Issues one step's descriptors into the named VMEM slot, returning the next free offset.

    Each pointer advance follows the descriptor that reads it, because the scalar unit does not interlock.
    """
    row = vmem_slot * _VMEM_SLOT_STRIDE
    for operand in operands:
        put(offset=at, d=builder.dma_load(slot=operand.dma_slot, vmem_address=operand.vmem_rows[step] + row,
                                          rows=operand.rows, base_sreg=operand.pointer,
                                          stride_sreg=_STRIDE_REGISTER))
        at += 1
        put(offset=at, s=builder.saddi(rd=operand.pointer, rs=operand.pointer, value=operand.advance))
        at += 1
    return at


def _emit_clear(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                rows: dict[str, tuple[int, ...]]) -> int:
    """Clears the resident output tiles from the clearing tile, returning the next free offset."""
    for group in range(_TOKEN_GROUPS):
        for vreg in range(_TILE_VREGS):
            put(offset=at + _TILE_VREGS * group + vreg,
                x=builder.vld(vreg=_TILE_VREGS * group + vreg, row=rows["zero"][0] + _NUM_SUBLANES * vreg))
    return at + _TOKEN_GROUPS * _TILE_VREGS


def _emit_drain(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                rows: dict[str, tuple[int, ...]]) -> int:
    """Stores the resident output tiles and advances the drain pointer past them."""
    for group in range(_TOKEN_GROUPS):
        for vreg in range(_TILE_VREGS):
            put(offset=at + _TILE_VREGS * group + vreg,
                x=builder.vst(vreg=_TILE_VREGS * group + vreg,
                              row=rows["output"][0] + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
    at += _TOKEN_GROUPS * _TILE_VREGS
    put(offset=at, d=builder.dma_store(slot=0, vmem_address=rows["output"][0], rows=_OUTPUT_ROWS,
                                       base_sreg=_DRAIN_POINTER, stride_sreg=_STRIDE_REGISTER))
    at += 1
    put(offset=at, f=builder.flush_slot(mask=1))
    at += 1
    put(offset=at, s=builder.saddi(rd=_DRAIN_POINTER, rs=_DRAIN_POINTER, value=_OUTPUT_ROWS * _ROW_BYTES))
    return at + 1


def _emit_block(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, compute_slot: int,
                operands: tuple[StagedOperand, ...], rows: dict[str, tuple[int, ...]]) -> int:
    """Emits one block, which accumulates out of `compute_slot` while the next block lands in the other.

    Each step's leading flush collects the descriptors issued one step earlier.
    """
    for step in range(_STEPS):
        put(offset=at, f=builder.flush_slot(mask=3))
        at += 1
        at = _emit_stage(put=put, builder=builder, at=at, step=step, vmem_slot=1 - compute_slot,
                         operands=operands)
        step_bundles = emit_gemm_step(activation_row=rows["activation"][step], weight_row=rows["weight"][step],
                                      close_loop=False)
        for offset, slots in step_bundles.items():
            put(offset=at + offset, **slots)
        at += max(step_bundles) + 1
    put(offset=at, f=builder.vmemswap())
    return at + 1


def _place_matrix_commands(*, put: Callable[..., None], builder: asm_module.AsmBuilder,
                           stage_next: Callable[..., int | None], close_loop: bool) -> None:
    """Places every push, pop and accumulate of one step, greedily against the matrix output FIFO's depth.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        stage_next: Returns the load riding under one push, or None where there is nothing left to stage.
        close_loop: Determines whether the final accumulate carries the enclosing loop's closing bundle.
    """
    depth, per_push, per_pop = asm_module.mxu_output_fifo()
    at = _PUSH_AT
    outstanding = 0
    popped = 0
    pending = -1

    def flush_accumulate() -> None:
        """Places a drained tile's adds, held back so they never split a run of pushes."""
        nonlocal at, pending
        if pending < 0:
            return
        last = pending == _TOKEN_GROUPS - 1
        for vreg in range(_TILE_VREGS):
            accumulator = _REGISTERS.accumulator + _TILE_VREGS * pending + vreg
            put(offset=at, e=builder.vadd(vd=accumulator, va=accumulator, vb=_REGISTERS.result + vreg))
            if close_loop and last and vreg == _TILE_VREGS - 1:
                put(offset=at, f=builder.loop_end())
            at += 1
        if not last:
            at += _ACCUMULATE_POP_GAP
        pending = -1

    def drain_one() -> None:
        """Pops the oldest un-drained result, marking a tile ready once all of its registers have landed."""
        nonlocal at, outstanding, popped, pending
        vreg = popped % _TILE_VREGS
        if vreg == 0:
            # The adds read the result registers this pop is about to overwrite.
            flush_accumulate()
        put(offset=at, m=builder.vmatpop(vreg=_REGISTERS.result + vreg))
        at += _POP_II
        outstanding -= per_pop
        popped += 1
        if vreg == _TILE_VREGS - 1:
            pending = popped // _TILE_VREGS - 1

    for group in range(_TOKEN_GROUPS):
        for vreg in range(_TILE_VREGS):
            while outstanding + per_push > depth:
                drain_one()
            slots = {"m": builder.vmatpush(vreg=_REGISTERS.activation + _TILE_VREGS * group + vreg)}
            staged = stage_next(group=group, vreg=vreg)
            if staged is not None:
                slots["x"] = staged
            put(offset=at, **slots)
            at += _PUSH_II
            outstanding += per_push
    while outstanding > 0:
        drain_one()
    flush_accumulate()


def _flatten(bundles: dict[int, dict[str, int]]) -> tuple[int, ...]:
    """Returns one bundle mapping as the contiguous words it places, zero filling the offsets it leaves."""
    words = []
    for offset in range(max(bundles) + 1):
        word = 0
        for slot_word in bundles.get(offset, {}).values():
            word |= slot_word
        words.append(word)
    return tuple(words)
