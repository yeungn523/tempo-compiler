"""Derives the tiled GEMM's bundle placement from the published ISA contract and the loop nest it walks.

Every placement constant that follows from the machine is computed from what `compiler/minitpuisa.td` publishes
through `isa_model.py`, and the two that remain written out are scheduling policy their own docstrings name as such.
"""

from __future__ import annotations

import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import asm as asm_module  # noqa: E402
from bf16 import (  # noqa: E402
    _BLOCK_DEPTH,
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
    _ZERO_TILE_WORDS,
)
from isa_model import MATRIX_OCCUPANCY, RTL_PARAMS  # noqa: E402
from layout import RegisterMap, VmemAllocator, register_map  # noqa: E402


_MATLOAD_AT: int = (_TILE_VREGS - 1) + RTL_PARAMS["WB_W_VLD"] + 1
"""Offset the weight load issues at, being the first cycle every weight register has landed, which is one cycle
past the last weight load's writeback reservation."""

_PUSH_AT: int = _MATLOAD_AT + MATRIX_OCCUPANCY["vmatload"] + 1
"""Offset the first activation push issues at, being the first cycle the matrix controller is free again. A
command holds the controller for its occupancy, so the next becomes legal one cycle later."""

_PUSH_II: int = MATRIX_OCCUPANCY["vmatpush"] + 1
"""Offsets between consecutive pushes, being the interval the controller's occupancy allows."""

_POP_II: int = 5
"""Offsets between consecutive pops, which is scheduling policy rather than a consequence of the ISA. Placing them
closer than the controller's occupancy states an intent to issue and leaves the real distance to `asm.schedule`."""

_ACCUMULATE_POP_GAP: int = 2
"""Offsets left after a group's accumulate before the next pop, which is scheduling policy. The adds hold the
result registers the following pop overwrites, so the gap keeps them clear of its write window."""

_ZERO_ROWS: int = _TILE
"""Rows the clearing tile occupies, which one descriptor fills before any column tile runs."""

_DRAIN_POINTER: int = 1
"""Scalar register holding the drain buffer's host address."""

_STRIDE_REGISTER: int = 3
"""Scalar register holding one DMEM row's byte stride, which every descriptor reads."""

_SCALAR_GAP: int = asm_module.W_SCALAR + 1
"""Bundles a scalar write must clear before a descriptor may read the register, the unit interlocking nothing."""

_MAX_SADDI: int = 0x7FFF
"""Widest displacement one scalar add carries, being the signed reach of its 16 bit immediate field."""

_REDUCTION_AXIS: str = "k"
"""Axis the contraction walks."""

_COLUMN_AXIS: str = "n"
"""Axis the output columns walk."""


@dataclass(frozen=True)
class LoopLevel:
    """Describes one loop of the emitted nest.

    Attributes:
        axis: Problem axis the loop walks, being "n" for output columns and "k" for the reduction. The token
            axis is absent, since the host issues one launch per token block.
        trips: Iterations the loop runs.
    """

    axis: str
    trips: int


@dataclass(frozen=True)
class Staging:
    """Describes what a host stages for one emitted image and what it reads back.

    The emitter settles these while placing the image, so a host takes them from here rather than deriving a
    second set that can disagree with it.

    Attributes:
        rows: Token rows one launch covers.
        columns: Output columns the product covers.
        output_tiles: Output tiles one block covers, being how the drain groups what it holds.
        blocks: Blocks the column loop walks, being the output tiles over the tiles one block covers.
        partials: Partial products each column tile leaves for the host to sum.
        drain_rows: VMEM rows one drain writes, being the resident tiles the accumulators hold.
    """

    rows: int
    columns: int
    output_tiles: int
    blocks: int
    partials: int
    drain_rows: int

    @property
    def column_tiles(self) -> int:
        """Returns the tiles the product covers, being the blocks it walks by the tiles each holds."""
        return self.blocks * self.output_tiles

    @property
    def drain_words(self) -> int:
        """Returns the words a launch drains, including the clearing tile the image leaves at the front."""
        return _ZERO_TILE_WORDS + self.blocks * self.partials * self.output_tiles * self.rows * _TILE

    @property
    def activation_slack(self) -> int:
        """Returns the bytes past the packed activation the trailing prefetch reads, which a host allocates.

        The block loop stages one block ahead, so the last iteration reads a block that is not there. The read is
        discarded, but the address must still be mapped.
        """
        return _STEPS * self.rows * _ROW_BYTES


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
        tiles: Tiles a step stages, each taking its own descriptor because DMEM does not hold them side by side.
        tile_stride: Bytes between one tile's region in DMEM and the next, which is nothing at a single tile.
    """

    name: str
    axes: frozenset[str]
    argument: int
    dma_slot: int
    vmem_rows: tuple[int, ...]
    rows: int
    pointer: int
    tiles: int = 1
    tile_stride: int = 0

    @property
    def advance(self) -> int:
        """Returns the bytes the pointer moves once its descriptor has read it."""
        return self.rows * _ROW_BYTES


def vmem_map(*, output_tiles: int = 1, token_groups: int = _TOKEN_GROUPS,
             spilled: int = 0) -> dict[str, tuple[int, ...]]:
    """Returns the first VMEM row of every region a launch stages, laid out end to end from row zero.

    Args:
        output_tiles: Output tiles one block covers, being the weight tiles a step stages and the tiles it drains.
        token_groups: Token groups one block covers, being the tiles its activation bank holds.
        spilled: Output tiles accumulating in VMEM rather than in registers, which own a region of their own.

    Returns:
        The clearing tile's row, each step's activation and weight rows, the output's row, and the spilled
        accumulators' row when a block has any.
    """
    vmem = VmemAllocator()
    rows = {"zero": (vmem.region("zero", _ZERO_ROWS).base,)}
    activation, weight = [], []
    for step in range(_STEPS):
        activation.append(vmem.region(f"activation {step}", token_groups * _TILE).base)
        weight.append(vmem.region(f"weight {step}", _TILE * output_tiles).base)
    rows["activation"] = tuple(activation)
    rows["weight"] = tuple(weight)
    rows["output"] = (vmem.region("output", token_groups * _TILE * output_tiles).base,)
    if spilled:
        rows["spill"] = (vmem.region("spill", token_groups * _TILE * spilled).base,)
    vmem.check_disjoint()
    return rows


def gemm_operands(*, rows: dict[str, tuple[int, ...]], output_tiles: int = 1, tile_stride: int = 0,
                  token_groups: int = _TOKEN_GROUPS) -> tuple[StagedOperand, ...]:
    """Returns every operand a launch stages, in the order their descriptors issue.

    Args:
        rows: Where each region sits, as `vmem_map` laid it out.
        output_tiles: Output tiles one block covers, being the weight tiles a step stages for it.
        tile_stride: Bytes between one weight tile's region in DMEM and the next.
        token_groups: Token groups one block covers, being the tiles its activation bank holds.

    Returns:
        The activation and the weight.
    """
    activation = StagedOperand(
        name="activation", axes=frozenset({_REDUCTION_AXIS}), argument=0, dma_slot=0,
        vmem_rows=rows["activation"], rows=token_groups * _TILE, pointer=0)
    weight = StagedOperand(
        name="weight", axes=frozenset({_REDUCTION_AXIS, _COLUMN_AXIS}), argument=1, dma_slot=1,
        vmem_rows=rows["weight"], rows=_TILE, pointer=2, tiles=output_tiles, tile_stride=tile_stride)
    return activation, weight


def staging_levels(*, nest: tuple[LoopLevel, ...],
                   operands: tuple[StagedOperand, ...]) -> dict[int, tuple[StagedOperand, ...]]:
    """Returns the operands re-staged at each loop level, keyed by that level's depth in the nest.

    Notes:
        An operand is re-staged at the innermost loop whose axis its address depends on, because every loop inside
        that one leaves the address unchanged and would re-read what VMEM already holds.

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
        An output tile is invariant to the reduction, so a column loop nested inside one keeps all of its tiles
        live at once while a column loop outside every reduction loop keeps one.

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


def gemm_step(*, activation_row: int, weight_row: int, close_loop: bool,
                   registers: RegisterMap | None = None, tile: int = 0,
                   spill_row: int | None = None) -> dict[int, dict[str, int]]:
    """Returns one accumulation step's bundles, keyed by their offset from the step's first bundle.

    Notes:
        One rule places the matrix commands. Push as early as the output FIFO allows and pop only when a further
        push would overflow it, because a pop stalls on the array while a push is what the array waits on.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.
        registers: Where the block's operands live, or None for the one tile wide block.
        tile: Which of the block's output tiles this step accumulates into, which names its accumulator set.
        spill_row: First VMEM row of this tile's accumulators when it accumulates in VMEM, or None when it holds
            them in registers.

    Returns:
        The step's bundles, keyed by offset, each holding the slot words that bundle issues.
    """
    if registers is None:
        registers = register_map(block_m=_BLOCK_TOKENS, block_n=_TILE)
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so a push and a load co-issue where both fit."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    for vreg in range(_TILE_VREGS):
        put(offset=vreg,
            x=builder.vld(vreg=registers.weight + vreg, row=weight_row + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(offset=_TILE_VREGS + vreg,
            x=builder.vld(vreg=registers.activation + vreg, row=activation_row + _NUM_SUBLANES * vreg))
    put(offset=_MATLOAD_AT, m=builder.vmatload(vreg=registers.weight))

    if spill_row is not None and registers.reload_hoisted:
        # Reads what earlier steps accumulated, which this step's results do not feed, so the load rides beside
        # the operand staging.
        for group in range(registers.token_groups):
            for vreg in range(_TILE_VREGS):
                put(offset=_TILE_VREGS * (2 + group) + vreg,
                    x=builder.vld(vreg=registers.scratch + _TILE_VREGS * group + vreg,
                                  row=spill_row + _TILE_ROWS * group + _NUM_SUBLANES * vreg))

    def stage_next(*, group: int, vreg: int) -> int | None:
        """Returns the load of the next token group's tile, or None once the last group is staged."""
        if group + 1 >= registers.token_groups:
            return None
        return builder.vld(vreg=registers.activation + _TILE_VREGS * (group + 1) + vreg,
                           row=activation_row + _TILE_ROWS * (group + 1) + _NUM_SUBLANES * vreg)

    _place_matrix_commands(put=put, builder=builder, stage_next=stage_next, close_loop=close_loop,
                           registers=registers, tile=tile, spill_row=spill_row)
    return bundles


def looped_gemm(*, column_tiles: int, groups: int, loop_blocks: int, fuse_groups: bool = False,
                block_m: int = _BLOCK_TOKENS, block_n: int = _TILE) -> tuple[dict[int, dict[str, int]], Staging]:
    """Returns the tiled GEMM's bundles and what a host stages for them, both from one derivation.

    Notes:
        Staging is overlapped onto compute by double buffering. A block computes out of one VMEM slot while the
        next block's operands land in the other, and `vmemswap` exchanges them once per block.

    Args:
        column_tiles: Output column tiles the product covers, being the outer loop's trip count.
        groups: Reduction groups each column tile accumulates, being the middle loop's trip count.
        loop_blocks: Accumulation blocks one group carries, which the two way unroll halves into a trip count.
        fuse_groups: Determines whether the accumulators drain once per column tile, summing the contraction on
            the device in BF16, rather than once per group for the host to sum in float32.
        block_m: Token rows one block covers, being the activation tiles it stages and pushes.
        block_n: Output columns one block covers, whose tiles accumulate together and share one staging of the
            activations.

    Returns:
        The image's bundles, keyed by offset, and the staging a host reads them back with.

    Raises:
        ValueError: If a group carries an odd number of blocks, which the two way unroll cannot halve, if the block
            does not divide the columns the product covers, if the register file does not hold its accumulators, or
            if the nest stages an operand anywhere but the innermost loop.
    """
    if loop_blocks % 2:
        message = (
            f"Unable to unroll a group of {loop_blocks} blocks two ways. Double buffering alternates the VMEM "
            f"slot every block and a descriptor names its slot outright, so a group must carry an even count."
        )
        raise ValueError(message)

    tiles = block_n // _TILE
    if block_n < _TILE or block_n % _TILE or column_tiles % tiles:
        message = (
            f"Unable to cover {column_tiles} column tiles in blocks of {block_n}. A block is a whole number of "
            f"{_TILE} wide tiles and the product is a whole number of blocks."
        )
        raise ValueError(message)
    registers = register_map(block_m=block_m, block_n=block_n)

    # Walks blocks rather than tiles, since a block's tiles share the activations staged for it.
    nest = (LoopLevel(axis=_COLUMN_AXIS, trips=column_tiles // tiles),
            LoopLevel(axis=_REDUCTION_AXIS, trips=groups),
            LoopLevel(axis=_REDUCTION_AXIS, trips=loop_blocks // 2))
    rows = vmem_map(output_tiles=tiles, token_groups=registers.token_groups, spilled=registers.spilled)
    # Measures the gap `pack_weights` leaves between tiles, being one tile's whole contraction.
    tile_stride = groups * loop_blocks * _BLOCK_DEPTH * _ROW_BYTES
    staged = _innermost_staging(nest=nest, operands=gemm_operands(
        rows=rows, output_tiles=tiles, tile_stride=tile_stride, token_groups=registers.token_groups),
        registers=registers)

    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    at = _emit_prologue(put=put, builder=builder, at=0, rows=rows, weight=staged[-1])

    put(offset=at, f=builder.lbegin(nest[0].trips))
    at += 1
    at = _emit_rewind(put=put, builder=builder, at=at, loop=nest[0], operands=staged)

    # Clears the accumulators once per column tile, so they carry every group and the contraction sums on
    # the device in BF16.
    if fuse_groups:
        at = _emit_clear(put=put, builder=builder, at=at, rows=rows, registers=registers)

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
        at = _emit_clear(put=put, builder=builder, at=at, rows=rows, registers=registers)

    put(offset=at, f=builder.lbegin(nest[2].trips))
    at += 1
    for compute_slot in range(2):
        at = _emit_block(put=put, builder=builder, at=at, compute_slot=compute_slot, operands=staged,
                         rows=rows, registers=registers)
    put(offset=at, f=builder.loop_end())
    at += 1

    # Returns both descriptor slots, which the group's trailing prefetch still owns, for the drain to use.
    put(offset=at, f=builder.flush_slot(mask=3))
    at += 1
    if not fuse_groups:
        at = _emit_drain(put=put, builder=builder, at=at, rows=rows, registers=registers)
    put(offset=at, f=builder.loop_end())
    at += 1
    if fuse_groups:
        at = _emit_drain(put=put, builder=builder, at=at, rows=rows, registers=registers)
    # Steps over the tiles the block covered past the first, which one iteration's own advance does not reach.
    for operand in staged:
        at = _emit_split_advance(put=put, builder=builder, at=at, register=operand.pointer,
                                 delta=operand.tile_stride * (operand.tiles - 1), gap=_SCALAR_GAP)
    put(offset=at, f=builder.loop_end())
    at += 1
    put(offset=at, f=builder.halt())
    # Drains a spilled tile once per bank, since the blocks alternate banks and each leaves two partials.
    held = registers.resident or registers.spilled
    staging = Staging(rows=block_m, columns=column_tiles * _TILE, output_tiles=tiles,
                      blocks=nest[0].trips,
                      partials=(1 if fuse_groups else groups) * (2 if registers.spilled else 1),
                      drain_rows=held * registers.token_groups * _TILE)
    return bundles, staging


def looped_gemm_stream(*, column_tiles: int, groups: int, loop_blocks: int, fuse_groups: bool = False,
                       block_m: int = _BLOCK_TOKENS,
                       block_n: int = _TILE) -> tuple[tuple[int, ...], Staging]:
    """Returns the tiled GEMM as the contiguous bundle words it places, zero filling the offsets it leaves.

    Args:
        column_tiles: Output column tiles the product covers.
        groups: Reduction groups each column tile accumulates.
        loop_blocks: Accumulation blocks one group carries.
        fuse_groups: Determines whether the accumulators drain once per column tile rather than once per group.
        block_m: Token rows one block covers.
        block_n: Output columns one block covers.

    Returns:
        The bundle words in placement order, and the staging a host reads them back with.
    """
    bundles, staging = looped_gemm(column_tiles=column_tiles, groups=groups, loop_blocks=loop_blocks,
                                   fuse_groups=fuse_groups, block_m=block_m, block_n=block_n)
    return _flatten(bundles), staging


def _innermost_staging(*, nest: tuple[LoopLevel, ...], operands: tuple[StagedOperand, ...],
                       registers: RegisterMap) -> tuple[StagedOperand, ...]:
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
    # Counts every tile the block accumulates, since a spilled one is held as much as a resident one.
    registers_held = registers.output_tiles
    tiles = resident_tiles(nest=nest)
    if tiles > registers_held:
        message = (
            f"Unable to walk a nest holding {tiles} output tiles at once. The register map carries "
            f"{registers_held}, so a column loop inside the contraction needs an allocator to widen it. "
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
        so its pointer returns to the launch argument. One whose address does walk it is left where it stopped.

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


def _emit_split_advance(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, register: int,
                        delta: int, gap: int) -> int:
    """Issues the scalar adds moving one pointer by the given displacement, returning the next free offset.

    Notes:
        One add reaches as far as its 16 bit immediate holds, so a displacement past that is split across
        several. The tiles a block covers and the depth of its group both grow the rewind its staging ends on.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        at: Offset the first add sits at.
        register: Scalar register holding the pointer.
        delta: Signed bytes the pointer moves.
        gap: Offsets between one add and the next.

    Returns:
        The next free offset.
    """
    remaining = delta
    while remaining:
        step = max(-_MAX_SADDI, min(_MAX_SADDI, remaining))
        put(offset=at, s=builder.saddi(rd=register, rs=register, value=step))
        at += gap
        remaining -= step
    return at


def _emit_stage(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, step: int,
                vmem_slot: int, operands: tuple[StagedOperand, ...]) -> int:
    """Issues one step's descriptors into the named VMEM slot, returning the next free offset.

    Each pointer advance follows the descriptor that reads it, because the scalar unit does not interlock.
    """
    row = vmem_slot * _VMEM_SLOT_STRIDE
    for operand in operands:
        # Issues one descriptor per tile, since DMEM holds them a contraction apart rather than side by side.
        for tile in range(operand.tiles):
            if tile:
                # Frees the slot before this tile reuses it, waiting on this operand's alone so the other
                # stays in flight.
                put(offset=at, f=builder.flush_slot(mask=1 << operand.dma_slot))
                at += 1
            put(offset=at, d=builder.dma_load(slot=operand.dma_slot,
                                              vmem_address=operand.vmem_rows[step] + row + _TILE * tile,
                                              rows=operand.rows, base_sreg=operand.pointer,
                                              stride_sreg=_STRIDE_REGISTER))
            at += 1
            walked = operand.tile_stride * (operand.tiles - 1)
            step_bytes = operand.tile_stride if tile + 1 < operand.tiles else operand.advance - walked
            at = _emit_split_advance(put=put, builder=builder, at=at, register=operand.pointer,
                                     delta=step_bytes, gap=1)
    return at


def _emit_clear(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                rows: dict[str, tuple[int, ...]], registers: RegisterMap) -> int:
    """Clears the output tiles from the clearing tile, returning the next free offset."""
    held = registers.resident * registers.token_groups
    for group in range(held):
        for vreg in range(_TILE_VREGS):
            put(offset=at + _TILE_VREGS * group + vreg,
                x=builder.vld(vreg=registers.accumulator + _TILE_VREGS * group + vreg,
                              row=rows["zero"][0] + _NUM_SUBLANES * vreg))
    at += held * _TILE_VREGS
    if registers.spilled:
        at = _emit_clear_spill(put=put, builder=builder, at=at, rows=rows, registers=registers)
    return at


def _emit_clear_spill(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                      rows: dict[str, tuple[int, ...]], registers: RegisterMap) -> int:
    """Zeroes the spilled accumulators in both VMEM banks, returning the next free offset.

    Notes:
        A block ends in `vmemswap`, so consecutive blocks fold into opposite banks. Compute reaches only the bank
        it is on, so both are cleared across a swap and the pair of swaps leaves the bank the blocks expect.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        at: Offset the clear starts at.
        rows: Where each region sits, as `vmem_map` laid it out.
        registers: Where the block's operands live.

    Returns:
        The next free offset.
    """
    for _ in range(2):
        for vreg in range(_TILE_VREGS):
            put(offset=at, x=builder.vld(vreg=registers.scratch + vreg,
                                         row=rows["zero"][0] + _NUM_SUBLANES * vreg))
            at += 1
        for group in range(registers.spilled * registers.token_groups):
            for vreg in range(_TILE_VREGS):
                put(offset=at, x=builder.vst(vreg=registers.scratch + vreg,
                                             row=rows["spill"][0] + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
                at += 1
        put(offset=at, f=builder.vmemswap())
        at += 1
    return at


def _emit_drain(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                rows: dict[str, tuple[int, ...]], registers: RegisterMap) -> int:
    """Stores the output tiles and advances the drain pointer past them."""
    if registers.spilled:
        return _emit_drain_spill(put=put, builder=builder, at=at, rows=rows, registers=registers)
    held = registers.resident * registers.token_groups
    for group in range(held):
        for vreg in range(_TILE_VREGS):
            put(offset=at + _TILE_VREGS * group + vreg,
                x=builder.vst(vreg=registers.accumulator + _TILE_VREGS * group + vreg,
                              row=rows["output"][0] + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
    at += held * _TILE_VREGS
    drained = held * _TILE
    put(offset=at, d=builder.dma_store(slot=0, vmem_address=rows["output"][0], rows=drained,
                                       base_sreg=_DRAIN_POINTER, stride_sreg=_STRIDE_REGISTER))
    at += 1
    put(offset=at, f=builder.flush_slot(mask=1))
    at += 1
    put(offset=at, s=builder.saddi(rd=_DRAIN_POINTER, rs=_DRAIN_POINTER, value=drained * _ROW_BYTES))
    return at + 1


def _emit_drain_spill(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                      rows: dict[str, tuple[int, ...]], registers: RegisterMap) -> int:
    """Stores both banks' spilled accumulators, returning the next free offset.

    Notes:
        The tiles already lie in VMEM, so a descriptor stores them where they are. Each bank holds half the
        blocks, so both drain and a descriptor names its slot outright to reach the bank compute is not on.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        at: Offset the drain starts at.
        rows: Where each region sits, as `vmem_map` laid it out.
        registers: Where the block's operands live.

    Returns:
        The next free offset.
    """
    drained = registers.spilled * registers.token_groups * _TILE
    for slot in range(2):
        put(offset=at, d=builder.dma_store(slot=0, rows=drained, base_sreg=_DRAIN_POINTER,
                                           stride_sreg=_STRIDE_REGISTER,
                                           vmem_address=rows["spill"][0] + slot * _VMEM_SLOT_STRIDE))
        at += 1
        put(offset=at, f=builder.flush_slot(mask=1))
        at += 1
        put(offset=at, s=builder.saddi(rd=_DRAIN_POINTER, rs=_DRAIN_POINTER, value=drained * _ROW_BYTES))
        at += 1
    return at


def _emit_block(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, compute_slot: int,
                operands: tuple[StagedOperand, ...], rows: dict[str, tuple[int, ...]],
                registers: RegisterMap) -> int:
    """Emits one block, which accumulates out of `compute_slot` while the next block lands in the other.

    Each step's leading flush collects the descriptors issued one step earlier.
    """
    for step in range(_STEPS):
        put(offset=at, f=builder.flush_slot(mask=3))
        at += 1
        at = _emit_stage(put=put, builder=builder, at=at, step=step, vmem_slot=1 - compute_slot,
                         operands=operands)
        # Emits one step per output tile, each folding into its own accumulator set.
        for tile in range(registers.output_tiles):
            spilled = tile - registers.resident
            spill_row = None if spilled < 0 else (
                rows["spill"][0] + _TILE_ROWS * spilled * registers.token_groups)
            step_bundles = gemm_step(activation_row=rows["activation"][step],
                                          weight_row=rows["weight"][step] + _TILE * tile,
                                          close_loop=False, registers=registers, tile=tile,
                                          spill_row=spill_row)
            for offset, slots in step_bundles.items():
                put(offset=at + offset, **slots)
            at += max(step_bundles) + 1
    put(offset=at, f=builder.vmemswap())
    return at + 1


def _place_matrix_commands(*, put: Callable[..., None], builder: asm_module.AsmBuilder,
                           stage_next: Callable[..., int | None], close_loop: bool,
                           registers: RegisterMap, tile: int, spill_row: int | None = None) -> None:
    """Places every push, pop and accumulate of one step, greedily against the matrix output FIFO's depth.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        stage_next: Returns the load riding under one push, or None where there is nothing left to stage.
        close_loop: Determines whether the final accumulate carries the enclosing loop's closing bundle.
        registers: Where the block's operands live.
        tile: Which of the block's output tiles the accumulates fold into.
        spill_row: First VMEM row of this tile's accumulators when it accumulates in VMEM, or None for a tile the
            registers hold.
    """
    accumulator_base = registers.accumulator + tile * registers.token_groups * _TILE_VREGS
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
        last = pending == registers.token_groups - 1
        closing = close_loop and last
        if spill_row is None:
            for vreg in range(_TILE_VREGS):
                accumulator = accumulator_base + _TILE_VREGS * pending + vreg
                put(offset=at, e=builder.vadd(vd=accumulator, va=accumulator, vb=registers.result + vreg))
                if closing and vreg == _TILE_VREGS - 1:
                    put(offset=at, f=builder.loop_end())
                at += 1
        else:
            # Keeps the runs apart, since a vst reads VREG port A and every add and push holds it while issuing.
            group_row = spill_row + _TILE_ROWS * pending
            scratch = registers.scratch + (_TILE_VREGS * pending if registers.reload_hoisted else 0)
            if not registers.reload_hoisted:
                for vreg in range(_TILE_VREGS):
                    put(offset=at, x=builder.vld(vreg=scratch + vreg, row=group_row + _NUM_SUBLANES * vreg))
                    at += 1
            for vreg in range(_TILE_VREGS):
                put(offset=at, e=builder.vadd(vd=scratch + vreg, va=scratch + vreg,
                                              vb=registers.result + vreg))
                at += 1
            for vreg in range(_TILE_VREGS):
                put(offset=at, x=builder.vst(vreg=scratch + vreg, row=group_row + _NUM_SUBLANES * vreg))
                if closing and vreg == _TILE_VREGS - 1:
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
            # Folds the pending tile first, since its adds read the registers this pop overwrites.
            flush_accumulate()
        put(offset=at, m=builder.vmatpop(vreg=registers.result + vreg))
        at += _POP_II
        outstanding -= per_pop
        popped += 1
        if vreg == _TILE_VREGS - 1:
            pending = popped // _TILE_VREGS - 1

    for group in range(registers.token_groups):
        for vreg in range(_TILE_VREGS):
            while outstanding + per_push > depth:
                drain_one()
            slots = {"m": builder.vmatpush(vreg=registers.activation + _TILE_VREGS * group + vreg)}
            staged = stage_next(group=group, vreg=vreg)
            if staged is not None:
                slots["x"] = staged
            put(offset=at, **slots)
            at += _PUSH_II
            outstanding += per_push
    while outstanding > 0:
        drain_one()
    flush_accumulate()


def _merge(*, bundles: dict[int, dict[str, int]], offset: int, slots: dict[str, int]) -> None:
    """Merges one offset's slot words into the image, refusing what a bundle cannot issue.

    Notes:
        A bundle issues one word per slot, so a merge naming one the offset already holds is a placement bug and
        the word it would replace leaves an image nothing downstream re-checks.

    Args:
        bundles: The image under construction, keyed by offset.
        offset: The bundle the words are merged into.
        slots: Slot words keyed by slot letter.

    Raises:
        ValueError: If the offset already carries one of the named slots, or if the merge leaves a descriptor
            co-issued with a compute slot.
    """
    bundle = bundles.setdefault(offset, {})
    repeated = sorted(set(bundle) & set(slots))
    if repeated:
        message = (
            f"Unable to merge the {sorted(slots)} slots into bundle {offset}, which already carries "
            f"{sorted(bundle)}. A bundle issues one word per slot, so {repeated} would replace what is placed "
            f"there and the replaced instruction would leave the image silently. Place the operation at an offset "
            f"whose {repeated} is free."
        )
        raise ValueError(message)
    asm_module.check_dma_exclusive({**bundle, **slots})
    bundle.update(slots)


def _flatten(bundles: dict[int, dict[str, int]]) -> tuple[int, ...]:
    """Returns one bundle mapping as the contiguous words it places, zero filling the offsets it leaves."""
    words = []
    for offset in range(max(bundles) + 1):
        word = 0
        for slot_word in bundles.get(offset, {}).values():
            word |= slot_word
        words.append(word)
    return tuple(words)
