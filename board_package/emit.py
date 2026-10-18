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
    _LN_ROW_OUT,
    _LN_ROW_X,
    _MAX_DESCRIPTOR_ROWS,
    _NUM_SUBLANES,
    _ROW_ADD_BIAS,
    _ROW_ADD_LEFT,
    _ROW_ADD_OUT,
    _ROW_ADD_RIGHT,
    _ROW_FLASH_BASE,
    _ROW_BYTES,
    _STEPS,
    _TILE,
    _TILE_ROWS,
    _TILE_VREGS,
    _TOKEN_GROUPS,
    _VMEM_ENTRIES_PER_LANE,
    _VMEM_SLOT_STRIDE,
    _VREG_ELEMENTS,
    _ZERO_TILE_BYTES,
    _ZERO_TILE_WORDS,
)
from isa_model import MATRIX_OCCUPANCY, RTL_PARAMS  # noqa: E402
from layout import (FlashRegisterMap, RegisterMap, VmemAllocator, flash_register_map,  # noqa: E402
                    register_map)


_MATLOAD_AT: int = (_TILE_VREGS - 1) + RTL_PARAMS["WB_W_VLD"] + 1
"""Offset the weight load issues at, being the first cycle every weight register has landed, which is one cycle
past the last weight load's writeback reservation."""

_PUSH_AT: int = _MATLOAD_AT + MATRIX_OCCUPANCY["vmatload"] + 1
"""Offset the first activation push issues at, being the first cycle the matrix controller is free again. A
command holds the controller for its occupancy, so the next becomes legal one cycle later."""

_GELU_AT: int = RTL_PARAMS["WB_W_VLD"] + 2
"""Offset the first activation issues at in an activating step, each one reading its tile two bundles past that
tile's load writeback."""

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

_POINTER_LATENCY: int = 2
"""Bundles a pointer advance is left before the branch reading it, matching the gap the key loop carries."""

_SCALAR_GAP: int = asm_module.W_SCALAR + 1
"""Bundles a scalar write must clear before a descriptor may read the register, the unit interlocking nothing."""

_MAX_SADDI: int = 0x7FFF
"""Widest displacement one scalar add carries, being the signed reach of its 16 bit immediate field."""

_REDUCTION_AXIS: str = "k"
"""Axis the contraction walks."""

_COLUMN_AXIS: str = "n"
"""Axis the output columns walk."""


_MATLOAD_HOLD: int = MATRIX_OCCUPANCY["vmatload"] + 1
"""Offsets a weight load holds the matrix controller, which is where a step's first push becomes legal."""

_TXIN_II: int = 2
"""Bundles between consecutive vtxin, being the spacing the board verified for a tile transpose."""

_TXIN_TO_TXOUT: int = 4
"""Bundles the first vtxout waits after the last vtxin. The hazard model covers vector and scalar registers, so
the staging array's own dependency is held by the schedule."""

_TXOUT_II: int = 6
"""Bundles between consecutive vtxout, being the spacing the board verified for a tile transpose."""

_VMEM_TURNAROUND: int = 8
"""Bundles an access waits after the one whose VMEM rows it reads, the hazard model covering registers alone."""

_SUBLANE_FOLDS: tuple[int, ...] = tuple(_NUM_SUBLANES >> (step + 1)
                                        for step in range(_NUM_SUBLANES.bit_length() - 1))
"""Sublanes each step of the cross-sublane fold reads over, halving until sublane zero holds the whole row."""

_SUBLANE_ZERO: int = 1
"""Sublane mask selecting the one sublane a folded row ends in."""

_FLASH_QUERY_POINTER: int = 0
"""Scalar register holding the queries, which an attention block stages once."""

_FLASH_OPERAND_POINTER: int = 1
"""Scalar register holding the keys and values, which advances one key block per trip."""

_FLASH_OUTPUT_POINTER: int = 2
"""Scalar register holding the attended output."""

_TILE_SHIFT: int = _TILE.bit_length() - 1
"""Shift the AGU walks a weight matrix by per iteration, one iteration reading one tile."""

_KEY_LOOP_LEVEL: int = 1
"""Loop level an attention block's key loop runs at, inside the one loop walking its query blocks or its heads."""

_RESIDENT_CONSTANTS: int = 0
"""Scalar register, and launch argument, holding a resident attention image's constants block."""

_RESIDENT_KEYS: int = 1
"""Scalar register walking the current head's keys and values through the fused tensor or the KV cache."""

_RESIDENT_CACHE: int = 1
"""Launch argument holding a decode image's KV cache."""

_RESIDENT_QUERIES: int = 2
"""Scalar register, and launch argument, holding the current head's queries, which its output overwrites."""

_COPY_SOURCE: int = 0
"""Scalar register, and launch argument, holding a strided copy's source."""

_COPY_DESTINATION: int = 2
"""Scalar register, and launch argument, holding a strided copy's destination."""

_ROW_COPY: int = 0
"""First VMEM row a strided copy passes its rows through."""

_RESIDENT_KEY_LEVEL: int = 3
"""Loop level a resident attention image's key loop runs at, inside the token block, head and query half loops."""

_TOKEN_BLOCK_SHIFT: int = _BLOCK_TOKENS.bit_length() - 1
"""Shift a walked load advances by per key token block, being the rows of two mask tiles."""

_POINTWISE_POINTERS: tuple[int, int, int] = (0, 1, 2)
"""Scalar registers holding a pointwise image's left, right and output pointers, each bound from the launch argument
of the same position."""

POINTWISE_LEFT: int = 0
"""VREG a pointwise body reads its left operand from."""

POINTWISE_RIGHT: int = 1
"""VREG a pointwise body reads its right operand from."""

_POINTWISE_TILE_SHIFT: int = (_MAX_DESCRIPTOR_ROWS // _BLOCK_TOKENS * _NUM_SUBLANES).bit_length() - 1
"""Shift the AGU walks a broadcast row by per chunk, a chunk spanning that many replicated rows of feature tiles."""

_ALU_KINDS: frozenset[str] = frozenset({"vadd", "vsub", "vmul", "vmax", "vmin"})
"""Vector operations the ALU runs, whose results land after its writeback."""

_ROW_POINTERS: tuple[int, int, int] = (0, 1, 2)
"""Scalar registers holding a row image's input, constants and output pointers, each bound from the launch argument
of the same position."""


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


@dataclass(frozen=True, slots=True)
class Load:
    """Reads one tile of a VMEM region into a register region.

    Attributes:
        into: Register region the tile lands in.
        region: VMEM region the tile is read from.
        tile: Which tile of that region.
        walked: Determines whether the key loop's counter advances the tile by one per trip, so each key block
            reads a tile of its own.
    """

    into: str
    region: str
    tile: int = 0
    walked: bool = False


@dataclass(frozen=True, slots=True)
class Store:
    """Writes a register region back to one tile of a VMEM region.

    Attributes:
        source: Register region written out.
        region: VMEM region the tile belongs to.
        tile: Which tile of that region.
    """

    source: str
    region: str
    tile: int = 0


@dataclass(frozen=True, slots=True)
class Product:
    """Streams one stationary tile through the array, leaving the product in the result region.

    Attributes:
        pushed: Register region streamed against the stationary tile.
        weight: Register region holding the stationary tile.
    """

    pushed: str
    weight: str


@dataclass(frozen=True, slots=True)
class Combine:
    """Applies one elementwise operation to whole register regions.

    Attributes:
        kind: The operation, being one of the vector ALU's mnemonics.
        into: Register region the result lands in.
        left: Register region of the left operand.
        right: Register region of the right operand.
    """

    kind: str
    into: str
    left: str
    right: str


@dataclass(frozen=True, slots=True)
class Transform:
    """Applies one single operand operation to a whole register region.

    Attributes:
        kind: The operation, being a move or one of the SFU's mnemonics.
        into: Register region the result lands in.
        source: Register region read.
    """

    kind: str
    into: str
    source: str


@dataclass(frozen=True, slots=True)
class RowReduce:
    """Reduces each row of a register region across its lanes, broadcast back along them.

    Attributes:
        kind: The operation the rows fold under, being vmax or vadd.
        source: Register region holding the tile.
        into: Register region the broadcast result lands in.
    """

    kind: str
    source: str
    into: str


@dataclass(frozen=True, slots=True)
class Accumulate:
    """Applies one elementwise operation to an output tile, wherever the register map placed it.

    A step list states the update, and the register map decides whether it runs in registers or through the
    scratch tile.

    Attributes:
        kind: The operation applied, being vmul for a rescale and vadd for a product.
        tile: Which output tile is updated.
        operand: Register region of the right operand.
        store: Determines whether a resident tile is written back to VMEM, which the drain needs.
    """

    kind: str
    tile: int
    operand: str
    store: bool = False


@dataclass(frozen=True, slots=True)
class VectorOp:
    """Describes one vector operation a pointwise body runs on every VREG it walks.

    Attributes:
        kind: The builder mnemonic the operation issues as, such as "vadd".
        into: The VREG the result lands in.
        sources: The VREGs the operation reads, in operand order.
    """

    kind: str
    into: int
    sources: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RowLoad:
    """Describes one VREG a row image loads from the regions it stages.

    Attributes:
        into: The VREG the load lands in.
        region: "input" for the token's own row, "row" for a per-feature row, or "scalar" for a broadcast constant.
        index: Which per-feature row or constant the load reads, zero for the input.
        tile: Which VREG of the row the load reads, zero for a constant.
    """

    into: int
    region: str
    index: int
    tile: int


@dataclass(frozen=True, slots=True)
class RowStore:
    """Describes one VREG of the output row a row image stores.

    Attributes:
        source: The VREG holding the result.
        tile: Which VREG of the output row it fills.
    """

    source: int
    tile: int


Step = Load | Store | Product | Combine | Transform | RowReduce | Accumulate
"""One operation of the vocabulary an image's body is written in. Aliased rather than declared with the `type`
statement, which the board's Python 3.11 interpreter does not carry."""


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
        partials: Partial products each column tile leaves for the host to sum, being two for a block accumulating
            in VMEM, whose consecutive blocks fold into opposite banks.
        drain_rows: VMEM rows one drain writes, being the resident tiles the accumulators hold.
        zero_tile_words: Words of clearing tile the drain opens with, which an image leaving none reports as zero.
    """

    rows: int
    columns: int
    output_tiles: int
    blocks: int
    partials: int
    drain_rows: int
    zero_tile_words: int = _ZERO_TILE_WORDS

    @property
    def column_tiles(self) -> int:
        """Returns the tiles the product covers, being the blocks it walks by the tiles each holds."""
        return self.blocks * self.output_tiles

    @property
    def drain_words(self) -> int:
        """Returns the words a launch drains, including the clearing tile the image leaves at the front."""
        return self.zero_tile_words + self.blocks * self.partials * self.output_tiles * self.rows * _TILE

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


@dataclass(slots=True)
class _Prefetch:
    """Hands one transfer's descriptors out a slot at a time, so each rides compute rather than blocking on it.

    Attributes:
        first_row: First VMEM row the transfer covers.
        rows: Rows the transfer moves.
        pointer: Scalar register holding the host address the transfer reads or writes.
        store: Determines whether the transfer writes host memory rather than reading it.
        moved: Rows the descriptors handed out so far cover.
        live: Descriptor slots still carrying one of them.
    """

    first_row: int
    rows: int
    pointer: int
    store: bool = False
    moved: int = 0
    live: tuple[int, ...] = ()

    @property
    def mask(self) -> int:
        """Returns the descriptor slots this transfer lands on, which is what a flush for it names.

        Returns:
            One bit per slot. Only a descriptor asserts a slot's done bit, so a flush naming a slot this never
            reaches would wait on one that never settles.
        """
        descriptors = -(-self.rows // _MAX_DESCRIPTOR_ROWS)
        return (1 << min(descriptors, asm_module.DMA_SLOT_COUNT)) - 1

    def issue(self, *, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int) -> int:
        """Issues one descriptor per slot, freeing each slot it reuses.

        Args:
            put: Merges slot words into the bundle at one offset.
            builder: Encodes each slot word.
            at: The bundle offset the first descriptor occupies.

        Returns:
            The next free bundle offset.
        """
        for slot in range(asm_module.DMA_SLOT_COUNT):
            if self.moved >= self.rows:
                break
            if slot in self.live:
                put(offset=at, f=builder.flush_slot(mask=1 << slot))
                at += 1
                self.live = tuple(ride for ride in self.live if ride != slot)
            at = self._descriptor(put=put, builder=builder, at=at, slot=slot)
        return at

    def finish(self, *, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, wait: bool) -> int:
        """Issues whatever the issue points had no room for, blocking on each, and waits for the rest.

        Args:
            put: Merges slot words into the bundle at one offset.
            builder: Encodes each slot word.
            at: The bundle offset the first descriptor occupies.
            wait: Determines whether the transfer is waited on here rather than by a later flush.

        Returns:
            The next free bundle offset.
        """
        while self.moved < self.rows:
            at = self.issue(put=put, builder=builder, at=at)
        if wait and self.live:
            put(offset=at, f=builder.flush_slot(mask=sum(1 << slot for slot in self.live)))
            at += 1
            self.live = ()
        return at

    def _descriptor(self, *, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int,
                    slot: int) -> int:
        """Places one descriptor on the given slot and the pointer advance that follows it."""
        span = min(self.rows - self.moved, _MAX_DESCRIPTOR_ROWS)
        issue = builder.dma_store if self.store else builder.dma_load
        put(offset=at, d=issue(slot=slot, vmem_address=self.first_row + self.moved, rows=span,
                               base_sreg=self.pointer, stride_sreg=_STRIDE_REGISTER))
        at += 1
        # The advance follows the descriptor that reads it, the scalar unit interlocking nothing.
        put(offset=at, s=builder.saddi(rd=self.pointer, rs=self.pointer, value=span * _ROW_BYTES))
        at += 1 + _SCALAR_GAP - 1
        self.moved += span
        self.live += (slot,)
        return at


def drain_descriptors(rows: int) -> tuple[int, int]:
    """Returns the rows one drain descriptor moves and how many of them a region of the given rows takes.

    Notes:
        A descriptor's row field encodes `_MAX_DESCRIPTOR_ROWS`, so a wider region is moved by several. The
        widest count dividing the region evenly is taken, which leaves every descriptor the same size and the
        region no larger than the rows it holds.

    Args:
        rows: Rows the region holds.

    Returns:
        The rows one descriptor moves and the descriptors the region takes.

    Raises:
        ValueError: If the region holds no rows.
    """
    if rows < 1:
        message = f"Unable to drain a region of {rows} rows. A region holds at least one."
        raise ValueError(message)
    for moved in range(min(rows, _MAX_DESCRIPTOR_ROWS), 0, -1):
        if not rows % moved:
            return moved, rows // moved
    return 1, rows


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
        # Reserves the tiles a step stages as the tiles its descriptors move, one apiece, so a block wider
        # than the row field encodes still lays out.
        weight.append(vmem.region(f"weight {step}", _TILE, tiles=output_tiles).base)
    rows["activation"] = tuple(activation)
    rows["weight"] = tuple(weight)
    moved, descriptors = drain_descriptors(token_groups * _TILE * output_tiles)
    rows["output"] = (vmem.region("output", moved, tiles=descriptors).base,)
    if spilled:
        moved, descriptors = drain_descriptors(token_groups * _TILE * spilled)
        rows["spill"] = (vmem.region("spill", moved, tiles=descriptors).base,)
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


def looped_gemm(*, column_tiles: int, groups: int, loop_blocks: int, activate: bool = False,
                block_m: int = _BLOCK_TOKENS,
                block_n: int = _TILE) -> tuple[dict[int, dict[str, int]], Staging]:
    """Returns the tiled GEMM's bundles and what a host stages for them, both from one derivation.

    Notes:
        Staging is overlapped onto compute by double buffering. A block computes out of one VMEM slot while the
        next block's operands land in the other, and `vmemswap` exchanges them once per block. The accumulators
        carry every group, so the contraction sums on the device in BF16 and drains once per column tile.

    Args:
        column_tiles: Output column tiles the product covers, being the outer loop's trip count.
        groups: Reduction groups each column tile accumulates, being the middle loop's trip count.
        loop_blocks: Accumulation blocks one group carries, which the two way unroll halves into a trip count.
        activate: Determines whether each staged activation tile passes through the GELU before it enters the
            array.
        block_m: Token rows one block covers, being the activation tiles it stages and pushes.
        block_n: Output columns one block covers, whose tiles accumulate together and share one staging of the
            activations.

    Returns:
        The image's bundles, keyed by offset, and the staging a host reads them back with.

    Raises:
        ValueError: If a group carries an odd number of blocks, which the two way unroll cannot halve, if the block
            does not divide the columns the product covers, if the register file does not hold its accumulators,
            if the nest stages an operand anywhere but the innermost loop, or if an activating block covers more
            than one resident output tile.
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
    if activate and (registers.output_tiles != 1 or registers.resident != 1):
        message = (
            f"Unable to activate a block {block_n} columns wide. An activating step rewrites its staged tiles in "
            f"place, so a block may cover only one resident {_TILE} wide output tile, but got "
            f"{registers.output_tiles}."
        )
        raise ValueError(message)

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

    put(offset=at, f=builder.lbegin(nest[2].trips))
    at += 1
    for compute_slot in range(2):
        at = _emit_block(put=put, builder=builder, at=at, compute_slot=compute_slot, operands=staged,
                         rows=rows, registers=registers, activate=activate)
    put(offset=at, f=builder.loop_end())
    at += 1

    # Returns both descriptor slots, which the group's trailing prefetch still owns, for the drain to use.
    put(offset=at, f=builder.flush_slot(mask=3))
    at += 1
    put(offset=at, f=builder.loop_end())
    at += 1
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
                      partials=2 if registers.spilled else 1,
                      drain_rows=held * registers.token_groups * _TILE)
    return bundles, staging


def looped_gemm_stream(*, column_tiles: int, groups: int, loop_blocks: int, activate: bool = False,
                       block_m: int = _BLOCK_TOKENS,
                       block_n: int = _TILE) -> tuple[tuple[int, ...], Staging]:
    """Returns the tiled GEMM as the contiguous bundle words it places, zero filling the offsets it leaves.

    Args:
        column_tiles: Output column tiles the product covers.
        groups: Reduction groups each column tile accumulates.
        loop_blocks: Accumulation blocks one group carries.
        activate: Determines whether each staged activation tile passes through the GELU before it enters the
            array.
        block_m: Token rows one block covers.
        block_n: Output columns one block covers.

    Returns:
        The bundle words in placement order, and the staging a host reads them back with.
    """
    bundles, staging = looped_gemm(column_tiles=column_tiles, groups=groups, loop_blocks=loop_blocks,
                                   activate=activate, block_m=block_m, block_n=block_n)
    return _flatten(bundles), staging


def classify_register_use(step: Step) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Returns the register regions one step reads, the ones it leaves a value in, and the ones it spends.

    Args:
        step: The step to describe.

    Returns:
        The regions read, the regions left holding a value, and the regions spent as working space.

    Raises:
        ValueError: If the step is not one the vocabulary carries.
    """
    empty: frozenset[str] = frozenset()
    if isinstance(step, Load):
        return empty, frozenset({step.into}), empty
    if isinstance(step, Store):
        return frozenset({step.source}), empty, empty
    if isinstance(step, Product):
        return frozenset({step.pushed, step.weight}), frozenset({"result"}), empty
    if isinstance(step, Combine):
        return frozenset({step.left, step.right}), frozenset({step.into}), empty
    if isinstance(step, Transform):
        return frozenset({step.source}), frozenset({step.into}), empty
    if isinstance(step, RowReduce):
        return frozenset({step.source}), frozenset({step.into}), frozenset({"staged", "running"})
    if isinstance(step, Accumulate):
        return frozenset({step.operand, "accumulator"}), frozenset({"accumulator"}), frozenset({"scratch"})
    message = f"Unable to describe {step}, which is not a step the vocabulary carries."
    raise ValueError(message)


def check_steps(body: tuple[Step, ...]) -> None:
    """Refuses a body that reads a region after another step spent it as working space.

    Args:
        body: The steps to check, in the order they run.

    Raises:
        ValueError: If a step reads a region another spent, with nothing writing it in between.
    """
    spent_at: dict[str, int] = {}
    for index, step in enumerate(body):
        reads, writes, spent = classify_register_use(step)
        for region in reads:
            if region in spent_at:
                message = (
                    f"Unable to run step {index}, {step}, which reads {region} after step {spent_at[region]} "
                    f"spent it as working space and nothing wrote it since. Read the value before the step "
                    f"spending that region runs."
                )
                raise ValueError(message)
        for region in writes:
            spent_at.pop(region, None)
        for region in spent:
            spent_at[region] = index


def emit_steps(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, body: tuple[Step, ...],
               registers: FlashRegisterMap, rows: dict[str, int], turnaround: int,
               key_level: int = _KEY_LOOP_LEVEL, walk_shift: int = _TILE_SHIFT) -> int:
    """Places every step of one body, resolving its regions through the register map.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the body starts at.
        body: The steps to place, in the order they run.
        registers: Where the block's tiles live.
        rows: First VMEM row of each staged region, keyed by what it stages.
        turnaround: Bundles an access waits after the one whose VMEM rows it reads.
        key_level: Loop level the key loop runs at, which a walked load reads its counter from.
        walk_shift: Shift a walked load applies to that counter, being the rows one trip advances it by.

    Returns:
        The next free bundle index.

    Raises:
        ValueError: If a step names an operation the vocabulary does not carry.
    """
    check_steps(body)

    def region(name: str) -> int:
        """Returns the first VREG of a named register region."""
        return getattr(registers, name)

    for step in body:
        if isinstance(step, Load):
            for vreg in range(_TILE_VREGS):
                put(offset=slot, x=builder.vld(region(step.into) + vreg,
                                               _tile_row(rows[step.region], step.tile, vreg),
                                               agu=step.walked, shift=walk_shift if step.walked else 0,
                                               level=key_level if step.walked else 0))
                slot += 1
        elif isinstance(step, Store):
            for vreg in range(_TILE_VREGS):
                put(offset=slot, x=builder.vst(region(step.source) + vreg,
                                               _tile_row(rows[step.region], step.tile, vreg)))
                slot += 1
        elif isinstance(step, Product):
            slot = _emit_flash_product(put, builder, slot, weight=region(step.weight),
                                       pushed=region(step.pushed), result=registers.result)
        elif isinstance(step, Combine):
            for vreg in range(_TILE_VREGS):
                put(offset=slot, e=getattr(builder, step.kind)(region(step.into) + vreg,
                                                               region(step.left) + vreg,
                                                               region(step.right) + vreg))
                slot += 1
        elif isinstance(step, Transform):
            for vreg in range(_TILE_VREGS):
                put(offset=slot, e=getattr(builder, step.kind)(region(step.into) + vreg,
                                                               region(step.source) + vreg))
                slot += 1
        elif isinstance(step, RowReduce):
            slot = _emit_row_reduce(put, builder, slot, source=_tile_registers(region(step.source)),
                                    transposed=_tile_registers(registers.staged),
                                    work=(registers.running, registers.running + 1),
                                    combine=getattr(builder, step.kind),
                                    destination=_tile_registers(region(step.into)),
                                    scratch_row=rows["scratch"], turnaround=turnaround)
        elif isinstance(step, Accumulate):
            slot = _emit_tile_update(put, builder, slot, tile=step.tile,
                                     registers=registers, operation=getattr(builder, step.kind),
                                     operand=region(step.operand), row=rows["output"],
                                     turnaround=turnaround, store=step.store)
        else:
            message = f"Unable to place {step}, which is not a step the vocabulary carries."
            raise ValueError(message)
    return slot

def flash_attention(*, key_blocks: int, queries: int = _TILE, head: int = _VREG_ELEMENTS,
                    turnaround: int = _VMEM_TURNAROUND, body: tuple[Step, ...],
                    epilogue: tuple[Step, ...], constants: int = 0,
                    row_blocks: int = 1, masked: bool = False) -> tuple[dict[int, dict[str, int]], Staging]:
    """Returns one attention block's bundles and what a host stages for them, the scores never reaching DMEM.

    Args:
        key_blocks: Blocks the loop walks, `_TILE` keys each.
        queries: Queries one block covers, a whole number of tiles.
        head: Width of the head, a whole number of tiles.
        turnaround: Bundles an access waits after the one whose VMEM rows it reads.
        body: The steps one trip of the key loop runs, which a compiler places from the kernel's own.
        epilogue: The steps the block runs once the loop has ended.
        constants: Tiles of stated numbers the kernel reads, staged ahead of the keys.
        row_blocks: Query blocks the image walks itself, each staging its own queries and draining its own
            output, which is one launch however many blocks the problem covers.
        masked: Determines whether each query block stages a mask tile per key block after its constants, which
            the body's walked loads read.

    Returns:
        The image's bundles, keyed by offset, and the staging a host reads them back with.

    Raises:
        ValueError: If the loop runs no blocks, if a block's operands overrun a descriptor, or if the queries
            cover more than one token group.
    """
    if key_blocks < 1 or row_blocks < 1:
        message = (
            f"Unable to attend over {key_blocks} key blocks for {row_blocks} query blocks. At least one of "
            f"each is needed."
        )
        raise ValueError(message)
    if 2 * head > _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to stage a head of {head} in one descriptor. A block reads its keys and values together, "
            f"which is {2 * head} rows against the {_MAX_DESCRIPTOR_ROWS} one descriptor moves."
        )
        raise ValueError(message)

    registers = flash_register_map(queries=queries, head=head)
    if registers.token_groups != 1:
        message = (
            f"Unable to attend over {queries} queries, which is {registers.token_groups} token groups. The row "
            f"reduction transposes one tile at a time, so a block covers {_TILE} queries."
        )
        raise ValueError(message)

    rows = _flash_vmem_map(head=head, constants=constants, masks=key_blocks if masked else 0)
    staged_rows = head + (2 + constants + (key_blocks if masked else 0)) * _TILE
    tiles = registers.output_tiles
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    slot = 0
    put(offset=slot, s=builder.smovi(rd=_STRIDE_REGISTER, value=_ROW_BYTES))
    slot += 1
    for register, argument in ((_FLASH_QUERY_POINTER, 0), (_FLASH_OPERAND_POINTER, 1),
                               (_FLASH_OUTPUT_POINTER, 2)):
        put(offset=slot, s=builder.smov_arg(rd=register, arg=argument))
        slot += 1
    slot += 1

    # Walks the query blocks here rather than over launches, each staging its own queries into the same rows.
    put(offset=slot, f=builder.lbegin(row_blocks))
    slot += 1
    # Stages the block's queries, its entering tiles, its constants and its mask tiles in as many descriptors as
    # their rows need, the query pointer walking each one's span.
    staged = 0
    while staged < staged_rows:
        span = min(staged_rows - staged, _MAX_DESCRIPTOR_ROWS)
        put(offset=slot, d=builder.dma_load(0, rows["query"] + staged, span, _FLASH_QUERY_POINTER,
                                            _STRIDE_REGISTER))
        slot += 1
        put(offset=slot, f=builder.flush_slot(mask=1))
        slot += 1
        staged += span
        if staged < staged_rows:
            put(offset=slot, s=builder.saddi(rd=_FLASH_QUERY_POINTER, rs=_FLASH_QUERY_POINTER,
                                             value=span * _ROW_BYTES))
            slot += _SCALAR_GAP
    walked_rows = staged - span

    slot = _emit_flash_entry(put, builder, slot, registers=registers, rows=rows)

    put(offset=slot, f=builder.lbegin(key_blocks))
    slot += 1
    put(offset=slot, d=builder.dma_load(0, rows["key"], 2 * head, _FLASH_OPERAND_POINTER, _STRIDE_REGISTER))
    slot += 1
    put(offset=slot, f=builder.flush_slot(mask=1))
    slot += 1

    slot = emit_steps(put, builder, slot, body=body, registers=registers, rows=rows,
                      turnaround=turnaround)

    put(offset=slot, s=builder.saddi(rd=_FLASH_OPERAND_POINTER, rs=_FLASH_OPERAND_POINTER,
                                     value=2 * head * _ROW_BYTES))
    slot += 1
    slot += 2
    put(offset=slot, f=builder.loop_end())
    slot += 1

    slot = emit_steps(put, builder, slot, registers=registers, rows=rows, turnaround=turnaround,
                      body=epilogue)
    slot += turnaround
    put(offset=slot, d=builder.dma_store(0, rows["output"], head, _FLASH_OUTPUT_POINTER, _STRIDE_REGISTER))
    slot += 1
    put(offset=slot, f=builder.flush_slot(mask=1))
    slot += 1

    # Advances to the next block's queries and output, and rewinds the keys every block reads from the start.
    advances = ((_FLASH_QUERY_POINTER, (staged_rows - walked_rows) * _ROW_BYTES),
                (_FLASH_OUTPUT_POINTER, head * _ROW_BYTES),
                (_FLASH_OPERAND_POINTER, -key_blocks * 2 * head * _ROW_BYTES))
    for register, value in advances:
        put(offset=slot, s=builder.saddi(rd=register, rs=register, value=value))
        slot += 1
    # Leaves the advances room to land, the next trip's descriptor reading the registers they write.
    slot += _POINTER_LATENCY
    put(offset=slot, f=builder.loop_end())
    slot += 1
    put(offset=slot, f=builder.halt())

    staging = Staging(rows=queries, columns=head, output_tiles=tiles, blocks=row_blocks, partials=1,
                      drain_rows=row_blocks * head, zero_tile_words=0)
    return bundles, staging

def resident_attention_stream(*, key_blocks: int, head: int, body: tuple[Step, ...], epilogue: tuple[Step, ...],
                              constants: int, heads: int, fused_tiles: int, cache_tiles: int | None = None,
                              turnaround: int = _VMEM_TURNAROUND) -> tuple[int, ...]:
    """Returns the image attending every head where the fused QKV tensor lies, as its bundle words.

    Notes:
        The launch binds the constants block as argument 0 and the fused tensor's first token block as argument 2.
        Each query block's output overwrites its queries, and each key tile is transposed in registers on its way
        to the array. The key loop walks key token blocks and unrolls their two halves, which keeps every pointer
        an affine walk that steps back to where its loop began. A decode image attends one query block, the new
        token in its first row, over a KV cache bound as argument 1.

    Args:
        key_blocks: Blocks of `_TILE` keys the sequence holds, which is also its query blocks.
        head: Width of the head, a whole number of tiles.
        body: The steps one key block runs.
        epilogue: The steps a query block runs once its keys are done.
        constants: Tiles of stated numbers the kernel reads, staged ahead of the mask tiles.
        heads: Heads the fused tensor carries.
        fused_tiles: Feature tiles one token of the fused tensor holds, being three times the model's.
        cache_tiles: Feature tiles one token of the KV cache holds, its keys and then its values, or None for an
            image whose keys and values are the fused tensor's own.

    Returns:
        The bundle words in placement order.

    Raises:
        ValueError: If the keys are not a whole number of token blocks, or the cache holds other than a key and a
            value per model feature.
    """
    halves = _BLOCK_TOKENS // _TILE
    model_tiles = fused_tiles // 3
    if key_blocks < halves or key_blocks % halves or cache_tiles not in (None, 2 * model_tiles):
        message = (
            f"Unable to attend over {key_blocks} key blocks from a cache of {cache_tiles} tiles. The image walks "
            f"whole token blocks of {halves} key blocks each, and a cache holds {2 * model_tiles} tiles, a key and "
            f"a value per model feature."
        )
        raise ValueError(message)
    key_token_blocks = key_blocks // halves
    query_halves = 1 if cache_tiles is not None else halves
    query_token_blocks = 1 if cache_tiles is not None else key_token_blocks
    registers = flash_register_map(queries=_TILE, head=head)
    tiles = registers.output_tiles
    rows = _flash_vmem_map(head=head, constants=constants, masks=key_blocks)
    tile_bytes = _BLOCK_TOKENS * _ROW_BYTES
    block_bytes = _ZERO_TILE_BYTES + fused_tiles * tile_bytes
    key_block_bytes = block_bytes if cache_tiles is None else _ZERO_TILE_BYTES + cache_tiles * tile_bytes
    model_bytes = model_tiles * tile_bytes
    half_bytes = _TILE * _ROW_BYTES
    head_bytes = tiles * tile_bytes
    constant_bytes = (2 + constants + key_blocks) * half_bytes
    walk = (tiles - 1) * tile_bytes
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    def advance(at: int, register: int, delta: int) -> int:
        """Moves one pointer by a displacement, returning the next free index."""
        return _emit_split_advance(put=put, builder=builder, at=at, register=register, delta=delta,
                                   gap=_SCALAR_GAP)

    put(offset=0, s=builder.smovi(rd=_STRIDE_REGISTER, value=_ROW_BYTES))
    put(offset=1, s=builder.smov_arg(rd=_RESIDENT_CONSTANTS, arg=_RESIDENT_CONSTANTS))
    put(offset=2, s=builder.smov_arg(rd=_RESIDENT_QUERIES, arg=_RESIDENT_QUERIES))
    # Reaches the keys past the fused tensor's queries, or at the start of a cache bound on its own.
    put(offset=3, s=builder.smov_arg(rd=_RESIDENT_KEYS,
                                     arg=_RESIDENT_QUERIES if cache_tiles is None else _RESIDENT_CACHE))
    slot = advance(3 + _SCALAR_GAP, _RESIDENT_KEYS, model_bytes if cache_tiles is None else 0)

    put(offset=slot, f=builder.lbegin(query_token_blocks))
    put(offset=slot + 1, f=builder.lbegin(heads))
    put(offset=slot + 2, f=builder.lbegin(query_halves))
    slot += 3
    slot = _emit_tile_copies(put, builder, slot, store=False, register=_RESIDENT_CONSTANTS,
                             first_row=rows["initial"], runs=constant_bytes // half_bytes, rows_each=_TILE,
                             step=half_bytes)
    # Steps past the last run, onto the next query half's constants.
    slot = advance(slot, _RESIDENT_CONSTANTS, half_bytes)
    slot = _emit_tile_copies(put, builder, slot, store=False, register=_RESIDENT_QUERIES, first_row=rows["query"],
                             runs=tiles, rows_each=_TILE, step=tile_bytes)
    slot = advance(slot, _RESIDENT_QUERIES, -walk)
    slot = _emit_flash_entry(put, builder, slot, registers=registers, rows=rows)

    put(offset=slot, f=builder.lbegin(key_token_blocks))
    slot += 1
    for half in range(halves):
        slot = _emit_tile_copies(put, builder, slot, store=False, register=_RESIDENT_KEYS, first_row=rows["incoming"],
                                 runs=tiles, rows_each=_TILE, step=tile_bytes)
        # Crosses from the head's keys to its values, which lie one model width of tiles further on.
        slot = advance(slot, _RESIDENT_KEYS, model_bytes - walk)
        slot = _emit_tile_copies(put, builder, slot, store=False, register=_RESIDENT_KEYS, first_row=rows["value"],
                                 runs=tiles, rows_each=_TILE, step=tile_bytes)
        crossed = model_bytes + walk
        following = (half_bytes - crossed if half < halves - 1
                     else key_block_bytes - (halves - 1) * half_bytes - crossed)
        slot = advance(slot, _RESIDENT_KEYS, following)
        slot = _emit_key_transposes(put, builder, slot, registers=registers, rows=rows, tiles=tiles)
        slot += turnaround
        # Reads this half's mask tiles, the key loop's counter stepping one token block of them per trip.
        halved = {**rows, "mask": rows["mask"] + half * _TILE}
        slot = emit_steps(put, builder, slot, body=body, registers=registers, rows=halved, turnaround=turnaround,
                          key_level=_RESIDENT_KEY_LEVEL, walk_shift=_TOKEN_BLOCK_SHIFT)
    slot += _POINTER_LATENCY
    put(offset=slot, f=builder.loop_end())
    slot += 1
    slot = advance(slot, _RESIDENT_KEYS, -key_token_blocks * key_block_bytes)

    slot = emit_steps(put, builder, slot, body=epilogue, registers=registers, rows=rows, turnaround=turnaround,
                      key_level=_RESIDENT_KEY_LEVEL)
    slot += turnaround
    slot = _emit_tile_copies(put, builder, slot, store=True, register=_RESIDENT_QUERIES, first_row=rows["output"],
                             runs=tiles, rows_each=_TILE, step=tile_bytes)
    # Moves on to the next query half, whose queries lie a half further down the same tiles.
    slot = advance(slot, _RESIDENT_QUERIES, half_bytes - walk)
    slot += _POINTER_LATENCY
    put(offset=slot, f=builder.loop_end())
    slot += 1

    # Moves on to the next head, rewinding the query halves and the constants every head shares.
    slot = advance(slot, _RESIDENT_QUERIES, head_bytes - query_halves * half_bytes)
    slot = advance(slot, _RESIDENT_KEYS, head_bytes)
    slot = advance(slot, _RESIDENT_CONSTANTS, -query_halves * constant_bytes)
    slot += _POINTER_LATENCY
    put(offset=slot, f=builder.loop_end())
    slot += 1

    # Moves on to the next token block, its queries one block on and its constants past this block's halves.
    slot = advance(slot, _RESIDENT_QUERIES, block_bytes - heads * head_bytes)
    slot = advance(slot, _RESIDENT_KEYS, -heads * head_bytes)
    slot = advance(slot, _RESIDENT_CONSTANTS, query_halves * constant_bytes)
    slot += _POINTER_LATENCY
    put(offset=slot, f=builder.loop_end())
    slot += 1
    put(offset=slot, f=builder.halt())
    return _flatten(bundles)


def strided_copy_stream(*, rows: int, stride: int) -> tuple[int, ...]:
    """Returns the image copying rows spaced `stride` bytes apart from argument 0 to argument 2, as its words.

    Args:
        rows: Rows to copy, each 32 bytes.
        stride: Bytes between one row and the next at both ends, being one row for a contiguous run.

    Returns:
        The bundle words in placement order.

    Raises:
        ValueError: If the copy moves no rows or the stride is not a whole number of rows.
    """
    if rows < 1 or stride < _ROW_BYTES or stride % _ROW_BYTES:
        message = (
            f"Unable to copy {rows} rows {stride} bytes apart. A copy moves at least one row, at a stride of a whole "
            f"number of {_ROW_BYTES}-byte rows."
        )
        raise ValueError(message)
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    put(offset=0, s=builder.smovi(rd=_STRIDE_REGISTER, value=stride))
    put(offset=1, s=builder.smov_arg(rd=_COPY_SOURCE, arg=_COPY_SOURCE))
    put(offset=2, s=builder.smov_arg(rd=_COPY_DESTINATION, arg=_COPY_DESTINATION))
    slot = 2 + _SCALAR_GAP
    copied = 0
    while copied < rows:
        span = min(rows - copied, _MAX_DESCRIPTOR_ROWS)
        put(offset=slot, d=builder.dma_load(slot=0, vmem_address=_ROW_COPY, rows=span, base_sreg=_COPY_SOURCE,
                                            stride_sreg=_STRIDE_REGISTER))
        put(offset=slot + 1, f=builder.flush_slot(mask=1))
        put(offset=slot + 2, d=builder.dma_store(slot=0, vmem_address=_ROW_COPY, rows=span,
                                                 base_sreg=_COPY_DESTINATION, stride_sreg=_STRIDE_REGISTER))
        put(offset=slot + 3, f=builder.flush_slot(mask=1))
        slot += 4
        copied += span
        if copied < rows:
            slot = _emit_split_advance(put=put, builder=builder, at=slot, register=_COPY_SOURCE,
                                       delta=span * stride, gap=_SCALAR_GAP)
            slot = _emit_split_advance(put=put, builder=builder, at=slot, register=_COPY_DESTINATION,
                                       delta=span * stride, gap=_SCALAR_GAP)
    put(offset=slot, f=builder.halt())
    return _flatten(bundles)


def flash_attention_stream(*, key_blocks: int, queries: int = _TILE, head: int = _VREG_ELEMENTS,
                           turnaround: int = _VMEM_TURNAROUND, body: tuple[Step, ...],
                           epilogue: tuple[Step, ...], constants: int = 0,
                           row_blocks: int = 1, masked: bool = False) -> tuple[tuple[int, ...], Staging]:
    """Returns one attention block as the contiguous bundle words it places.

    Args:
        key_blocks: Blocks the loop walks, `_TILE` keys each.
        queries: Queries one block covers.
        head: Width of the head.
        turnaround: Bundles an access waits after the one whose VMEM rows it reads.
        body: The steps one trip of the key loop runs, which a compiler places from the kernel's own.
        epilogue: The steps the block runs once the loop has ended.
        constants: Tiles of stated numbers the kernel reads, staged ahead of the keys.
        row_blocks: Query blocks the image walks itself, one launch covering all of them.
        masked: Determines whether each query block stages a mask tile per key block.

    Returns:
        The bundle words in placement order, and the staging a host reads them back with.
    """
    bundles, staging = flash_attention(key_blocks=key_blocks, queries=queries, head=head,
                                       turnaround=turnaround, body=body, epilogue=epilogue,
                                       constants=constants, row_blocks=row_blocks, masked=masked)
    return _flatten(bundles), staging


def pointwise(*, chunks: int, broadcast_right: bool, body: tuple[VectorOp, ...]) -> dict[int, dict[str, int]]:
    """Returns the bundles of a pointwise image, which runs `body` on every VREG of one token block.

    Notes:
        Each chunk moves `_MAX_DESCRIPTOR_ROWS` rows of every operand into VMEM, runs the body one VREG at a time, and
        moves the result back out. A broadcast right operand is staged once, its feature tiles replicated across the
        sublanes, and each chunk reads its own tiles through the AGU. The body reuses its registers on every VREG,
        so each operation issues once its operands have landed and the result is stored before the next VREG loads.

    Args:
        chunks: Descriptor chunks one token block spans.
        broadcast_right: Determines whether the right operand is one feature row broadcast down the tokens.
        body: The operations one VREG runs, reading the left operand from `POINTWISE_LEFT` and the right one from
            `POINTWISE_RIGHT`, and leaving the result in the last operation's `into`.

    Returns:
        The image's bundles, keyed by offset.

    Raises:
        ValueError: If the chunk count is not positive or the body runs no operation.
    """
    if chunks < 1 or not body:
        message = (
            f"Unable to emit a pointwise image over {chunks} chunks running {len(body)} operations. It needs at "
            f"least one chunk and one operation."
        )
        raise ValueError(message)

    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so co-issue reads at the call site."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    left, right, output = _POINTWISE_POINTERS
    put(offset=0, s=builder.smovi(rd=_STRIDE_REGISTER, value=_ROW_BYTES))
    for at, pointer in enumerate(_POINTWISE_POINTERS, start=1):
        put(offset=at, s=builder.smov_arg(rd=pointer, arg=pointer))
    at = len(_POINTWISE_POINTERS) + _SCALAR_GAP

    if broadcast_right:
        rows = chunks * _MAX_DESCRIPTOR_ROWS // _BLOCK_TOKENS * _NUM_SUBLANES
        staged = 0
        while staged < rows:
            span = min(rows - staged, _MAX_DESCRIPTOR_ROWS)
            put(offset=at, d=builder.dma_load(slot=1, vmem_address=_ROW_ADD_BIAS + staged, rows=span,
                                              base_sreg=right, stride_sreg=_STRIDE_REGISTER))
            put(offset=at + 1, f=builder.flush_slot(mask=2))
            at += 2
            staged += span
            if staged < rows:
                at = _emit_split_advance(put=put, builder=builder, at=at, register=right,
                                         delta=span * _ROW_BYTES, gap=_SCALAR_GAP)

    put(offset=at, f=builder.lbegin(chunks))
    put(offset=at + 1, d=builder.dma_load(slot=0, vmem_address=_ROW_ADD_LEFT, rows=_MAX_DESCRIPTOR_ROWS,
                                          base_sreg=left, stride_sreg=_STRIDE_REGISTER))
    at += 2
    if not broadcast_right:
        put(offset=at, d=builder.dma_load(slot=1, vmem_address=_ROW_ADD_RIGHT, rows=_MAX_DESCRIPTOR_ROWS,
                                          base_sreg=right, stride_sreg=_STRIDE_REGISTER))
        at += 1
    put(offset=at, f=builder.flush_slot(mask=1 if broadcast_right else 3))
    at += 1

    load_latency = RTL_PARAMS["WB_W_VLD"] + 1
    vregs_per_tile = _BLOCK_TOKENS // _NUM_SUBLANES
    for vreg in range(_MAX_DESCRIPTOR_ROWS // _NUM_SUBLANES):
        row = _NUM_SUBLANES * vreg
        put(offset=at, x=builder.vld(vreg=POINTWISE_LEFT, row=_ROW_ADD_LEFT + row))
        if broadcast_right:
            right_load = builder.vld(vreg=POINTWISE_RIGHT, row=_ROW_ADD_BIAS + vreg // vregs_per_tile * _NUM_SUBLANES,
                                     agu=True, shift=_POINTWISE_TILE_SHIFT, level=0)
        else:
            right_load = builder.vld(vreg=POINTWISE_RIGHT, row=_ROW_ADD_RIGHT + row)
        put(offset=at + 1, x=right_load)
        at = _emit_vector_body(put=put, builder=builder,
                               ready={POINTWISE_LEFT: at + load_latency, POINTWISE_RIGHT: at + 1 + load_latency},
                               body=body)
        put(offset=at, x=builder.vst(vreg=body[-1].into, row=_ROW_ADD_OUT + row))
        at += 1

    put(offset=at, d=builder.dma_store(slot=0, vmem_address=_ROW_ADD_OUT, rows=_MAX_DESCRIPTOR_ROWS,
                                       base_sreg=output, stride_sreg=_STRIDE_REGISTER))
    put(offset=at + 1, f=builder.flush_slot(mask=1))
    at += 2
    advanced = (left, output) if broadcast_right else (left, right, output)
    for pointer in advanced:
        put(offset=at, s=builder.saddi(rd=pointer, rs=pointer, value=_MAX_DESCRIPTOR_ROWS * _ROW_BYTES))
        at += 1
    # Closes the chunk loop on the bundle carrying the last pointer advance, as the hand schedule does.
    put(offset=at - 1, f=builder.loop_end())
    put(offset=at, f=builder.halt())
    return bundles


def pointwise_stream(*, chunks: int, broadcast_right: bool, body: tuple[VectorOp, ...]) -> tuple[int, ...]:
    """Returns a pointwise image as the contiguous bundle words it places.

    Args:
        chunks: Descriptor chunks one token block spans.
        broadcast_right: Determines whether the right operand is one feature row broadcast down the tokens.
        body: The operations one VREG runs.

    Returns:
        The bundle words in placement order.
    """
    return _flatten(pointwise(chunks=chunks, broadcast_right=broadcast_right, body=body))


def row_stream(*, features: int, rows: int, scalars: int, prologue: tuple[RowLoad, ...],
               body: tuple[RowLoad | VectorOp | RowStore, ...]) -> tuple[int, ...]:
    """Returns a row image, which runs `body` once per token of one token block, as one operation per bundle.

    Notes:
        The constants block, holding the per-feature rows and then the broadcast constants, is staged once and the
        prologue's loads are taken from it before the token loop. Each trip gathers one token's feature tiles, which
        lie a token block of rows apart in DMEM, into contiguous VMEM rows through the descriptor stride, so every
        VREG of the row holds one token's features. The body is emitted unplaced, leaving co-issue to the packing
        pass that every operation's register dependencies drive.

    Args:
        features: Features one token row holds.
        rows: Per-feature rows the constants block opens with.
        scalars: Broadcast constants the constants block closes with, one VREG each.
        prologue: The constant loads run once, before the token loop.
        body: The operations one token runs.

    Returns:
        The bundle words in program order.

    Raises:
        ValueError: If a row is not a whole number of VREGs, or the staged regions do not fit beneath the output.
    """
    width_rows = features // _TILE
    if features <= 0 or features % _VREG_ELEMENTS or width_rows > _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to emit a row image over {features} features. `vredsum` reduces one VREG, so a row must be a "
            f"whole number of {_VREG_ELEMENTS} features, and one descriptor gathers at most {_MAX_DESCRIPTOR_ROWS} "
            f"rows of it."
        )
        raise ValueError(message)
    constants_row = _LN_ROW_X + width_rows
    constants_rows = rows * width_rows + scalars * _NUM_SUBLANES
    if constants_row + constants_rows > _LN_ROW_OUT or _LN_ROW_OUT + width_rows > _VMEM_ENTRIES_PER_LANE:
        message = (
            f"Unable to place a row image's regions. The token's row and the constants occupy VMEM rows "
            f"[{_LN_ROW_X}:{constants_row + constants_rows}), which must end by the output at row {_LN_ROW_OUT}."
        )
        raise ValueError(message)

    builder = asm_module.AsmBuilder()
    words: list[dict[str, int]] = []

    def emit(**slots: int) -> None:
        """Appends one bundle carrying the given slot words."""
        words.append(slots)

    def region_row(load: RowLoad) -> int:
        """Returns the VMEM row one load reads."""
        if load.region == "input":
            return _LN_ROW_X + _NUM_SUBLANES * load.tile
        if load.region == "row":
            return constants_row + load.index * width_rows + _NUM_SUBLANES * load.tile
        return constants_row + rows * width_rows + load.index * _NUM_SUBLANES

    source, constants, output = _ROW_POINTERS
    emit(s=builder.smovi(rd=_STRIDE_REGISTER, value=_ROW_BYTES))
    for pointer in _ROW_POINTERS:
        emit(s=builder.smov_arg(rd=pointer, arg=pointer))
    for _ in range(_SCALAR_GAP - 1):
        emit()
    staged = 0
    while staged < constants_rows:
        span = min(constants_rows - staged, _MAX_DESCRIPTOR_ROWS)
        emit(d=builder.dma_load(slot=0, vmem_address=constants_row + staged, rows=span, base_sreg=constants,
                                stride_sreg=_STRIDE_REGISTER))
        emit(f=builder.flush_slot(mask=1))
        staged += span
        if staged < constants_rows:
            emit(s=builder.saddi(rd=constants, rs=constants, value=span * _ROW_BYTES))
            for _ in range(_SCALAR_GAP - 1):
                emit()
    # Strides the gather by one feature tile of the packed block, so the descriptor reads a token's tiles in turn.
    emit(s=builder.smovi(rd=_STRIDE_REGISTER, value=_BLOCK_TOKENS * _ROW_BYTES))
    for _ in range(_SCALAR_GAP - 1):
        emit()
    for load in prologue:
        emit(x=builder.vld(vreg=load.into, row=region_row(load)))

    emit(f=builder.lbegin(_BLOCK_TOKENS))
    emit(d=builder.dma_load(slot=0, vmem_address=_LN_ROW_X, rows=width_rows, base_sreg=source,
                            stride_sreg=_STRIDE_REGISTER))
    emit(s=builder.saddi(rd=source, rs=source, value=_ROW_BYTES))
    emit(f=builder.flush_slot(mask=1))
    for operation in body:
        if isinstance(operation, RowLoad):
            emit(x=builder.vld(vreg=operation.into, row=region_row(operation)))
        elif isinstance(operation, RowStore):
            emit(x=builder.vst(vreg=operation.source, row=_LN_ROW_OUT + _NUM_SUBLANES * operation.tile))
        else:
            names = ("va", "vb")[:len(operation.sources)]
            emit(e=getattr(builder, operation.kind)(vd=operation.into,
                                                    **dict(zip(names, operation.sources, strict=True))))
    emit(d=builder.dma_store(slot=0, vmem_address=_LN_ROW_OUT, rows=width_rows, base_sreg=output,
                             stride_sreg=_STRIDE_REGISTER))
    emit(f=builder.flush_slot(mask=1))
    emit(s=builder.saddi(rd=output, rs=output, value=_ROW_BYTES))
    emit(f=builder.loop_end())
    emit(f=builder.halt())
    return _flatten(dict(enumerate(words)))


def _emit_vector_body(*, put: Callable[..., None], builder: asm_module.AsmBuilder, ready: dict[int, int],
                      body: tuple[VectorOp, ...]) -> int:
    """Places a pointwise body's operations in order, each once its operands have landed.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        ready: Offset each loaded VREG lands at, which the body's own results extend.
        body: The operations to place.

    Returns:
        The offset the last result lands at, where its store may issue.
    """
    landed = dict(ready)
    for operation in body:
        at = max(landed[source] for source in operation.sources)
        names = ("va", "vb")[:len(operation.sources)]
        put(offset=at, e=getattr(builder, operation.kind)(vd=operation.into,
                                                          **dict(zip(names, operation.sources, strict=True))))
        writeback = RTL_PARAMS["WB_W_ALU"] if operation.kind in _ALU_KINDS else RTL_PARAMS["WB_W_SFU"]
        landed[operation.into] = at + writeback + 1
    return landed[body[-1].into]


def _emit_key_transposes(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *,
                         registers: FlashRegisterMap, rows: dict[str, int], tiles: int) -> int:
    """Transposes each incoming key tile into the stationary tile the score product reads, returning the next index.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the first load occupies.
        registers: Where the block's tiles live, whose shared span holds nothing between key blocks.
        rows: First VMEM row of each staged region, keyed by what it stages.
        tiles: Tiles the head covers.

    Returns:
        The next free bundle index, the last store issued.
    """
    source, destination = _tile_registers(registers.scores), _tile_registers(registers.staged)
    for tile in range(tiles):
        for vreg, register in enumerate(source):
            put(offset=slot, x=builder.vld(register, _tile_row(rows["incoming"], tile, vreg)))
            slot += 1
        slot = _emit_transpose(put, builder, slot, source=source, destination=destination)
        for vreg, register in enumerate(destination):
            put(offset=slot, x=builder.vst(register, _tile_row(rows["key"], tile, vreg)))
            slot += 1
    return slot


def _emit_flash_entry(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *,
                      registers: FlashRegisterMap, rows: dict[str, int]) -> int:
    """Enters one query block's running statistics and clears its accumulator, returning the next free index.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the first load occupies.
        registers: Where the block's tiles live.
        rows: First VMEM row of each staged region, keyed by what it stages.

    Returns:
        The next free bundle index.
    """
    maximum, total = _tile_registers(registers.row_max), _tile_registers(registers.row_sum)
    # Enters the maximum below every score, so the first rescale multiplies by zero.
    for vreg in range(_TILE_VREGS):
        put(offset=slot, x=builder.vld(maximum[vreg], rows["initial"] + _NUM_SUBLANES * vreg))
        slot += 1
        put(offset=slot, x=builder.vld(total[vreg], rows["zero"] + _NUM_SUBLANES * vreg))
        slot += 1
        if registers.spilled:
            put(offset=slot, x=builder.vld(registers.scratch + vreg, rows["zero"] + _NUM_SUBLANES * vreg))
            slot += 1
    for tile in range(registers.output_tiles):
        for vreg in range(_TILE_VREGS):
            cleared = (builder.vld(registers.accumulator + tile * _TILE_VREGS + vreg,
                                   rows["zero"] + _NUM_SUBLANES * vreg) if tile < registers.resident
                       else builder.vst(registers.scratch + vreg, _tile_row(rows["output"], tile, vreg)))
            put(offset=slot, x=cleared)
            slot += 1
    return slot


def _emit_tile_copies(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, store: bool,
                      register: int, first_row: int, runs: int, rows_each: int, step: int) -> int:
    """Copies runs of contiguous rows between VMEM and DMEM, each run's DMEM address `step` bytes past the last.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the first descriptor occupies.
        store: Determines whether the runs move from VMEM to DMEM, rather than from DMEM to VMEM.
        register: Scalar register holding the first run's DMEM address, which the copies leave on the last run's.
        first_row: VMEM row the first run occupies, the rest following it.
        runs: Runs to copy.
        rows_each: Rows one run covers.
        step: Bytes between one run's DMEM address and the next.

    Returns:
        The next free bundle index, every copy having landed.
    """
    copy = builder.dma_store if store else builder.dma_load
    for run in range(runs):
        put(offset=slot, d=copy(run % 2, first_row + run * rows_each, rows_each, register, _STRIDE_REGISTER))
        slot += 1
        # Waits on the pair of slots a descriptor alternates between, which is what one flush mask covers.
        if run % 2 or run == runs - 1:
            put(offset=slot, f=builder.flush_slot(mask=3 if run % 2 else 1))
            slot += 1
        if run < runs - 1:
            slot = _emit_split_advance(put=put, builder=builder, at=slot, register=register, delta=step,
                                       gap=_SCALAR_GAP)
    return slot


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
    moved, descriptors = drain_descriptors(held * _TILE)
    for descriptor in range(descriptors):
        put(offset=at, d=builder.dma_store(slot=0, vmem_address=rows["output"][0] + moved * descriptor,
                                           rows=moved, base_sreg=_DRAIN_POINTER,
                                           stride_sreg=_STRIDE_REGISTER))
        at += 1
        put(offset=at, f=builder.flush_slot(mask=1))
        at += 1
        at = _emit_split_advance(put=put, builder=builder, at=at, register=_DRAIN_POINTER,
                                 delta=moved * _ROW_BYTES, gap=1)
    return at


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
    moved, descriptors = drain_descriptors(registers.spilled * registers.token_groups * _TILE)
    for slot in range(2):
        for descriptor in range(descriptors):
            put(offset=at, d=builder.dma_store(slot=0, rows=moved, base_sreg=_DRAIN_POINTER,
                                               stride_sreg=_STRIDE_REGISTER,
                                               vmem_address=rows["spill"][0] + moved * descriptor
                                               + slot * _VMEM_SLOT_STRIDE))
            at += 1
            put(offset=at, f=builder.flush_slot(mask=1))
            at += 1
            at = _emit_split_advance(put=put, builder=builder, at=at, register=_DRAIN_POINTER,
                                     delta=moved * _ROW_BYTES, gap=1)
    return at


def _gelu_gemm_step(*, activation_row: int, weight_row: int, close_loop: bool,
                    registers: RegisterMap) -> dict[int, dict[str, int]]:
    """Returns one accumulation step's bundles with every staged activation tile passed through the GELU first.

    Notes:
        Every activation precedes the first push. VREG read port A is one physical port whose static-priority mux
        the matrix controller wins (``vpu.sv`` ``effective_raddr_a``), so a vgelu issued while a vmatpush streams
        reads the pushed register in place of its own operand. The hazard model does not see that port, so the
        offsets here carry the ordering themselves.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.
        registers: Where the block's operands live, which must hold one resident output tile.

    Returns:
        The step's bundles, keyed by offset, each holding the slot words that bundle issues.
    """
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so a push and a load co-issue where both fit."""
        _merge(bundles=bundles, offset=offset, slots=slots)

    staged = registers.token_groups * _TILE_VREGS
    for index in range(staged):
        group, vreg = divmod(index, _TILE_VREGS)
        put(offset=index, x=builder.vld(vreg=registers.activation + index,
                                        row=activation_row + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(offset=staged + vreg,
            x=builder.vld(vreg=registers.weight + vreg, row=weight_row + _NUM_SUBLANES * vreg))
    for index in range(staged):
        put(offset=_GELU_AT + index,
            e=builder.vgelu(vd=registers.activation + index, va=registers.activation + index))
    # Trails the weight loads, which start after the activation loads, and stays past the last vgelu's port A read.
    matload_at = max(staged + _MATLOAD_AT, _GELU_AT + staged)
    put(offset=matload_at, m=builder.vmatload(vreg=registers.weight))

    _place_matrix_commands(put=put, builder=builder, stage_next=lambda *, group, vreg: None,
                           close_loop=close_loop, registers=registers, tile=0,
                           origin=matload_at + _MATLOAD_HOLD)
    return bundles


def _emit_block(*, put: Callable[..., None], builder: asm_module.AsmBuilder, at: int, compute_slot: int,
                operands: tuple[StagedOperand, ...], rows: dict[str, tuple[int, ...]],
                registers: RegisterMap, activate: bool = False) -> int:
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
            if activate:
                step_bundles = _gelu_gemm_step(activation_row=rows["activation"][step],
                                               weight_row=rows["weight"][step], close_loop=False,
                                               registers=registers)
            else:
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
                           registers: RegisterMap, tile: int, spill_row: int | None = None,
                           origin: int = _PUSH_AT, pushed: tuple[int, ...] | None = None,
                           accumulators: tuple[int, ...] | None = None) -> None:
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
        origin: Offset the first push issues at, which a weight load read before the step opens brings forward.
        pushed: First register of each token group's pushed tile, for a product reading tiles the step did not
            stage.
        accumulators: First register of each token group's accumulator, for a product folding into tiles the
            block's own accumulators do not name.
    """
    groups = registers.token_groups
    pushed = pushed or tuple(registers.activation + _TILE_VREGS * group for group in range(groups))
    accumulators = accumulators or tuple(
        registers.accumulator + tile * groups * _TILE_VREGS + _TILE_VREGS * group for group in range(groups))
    depth, per_push, per_pop = asm_module.mxu_output_fifo()
    at = origin
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
                accumulator = accumulators[pending] + vreg
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
            slots = {"m": builder.vmatpush(vreg=pushed[group] + vreg)}
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


def _tile_registers(base: int) -> tuple[int, ...]:
    """Returns the VREGs one tile occupies, counting from the given register."""
    return tuple(base + vreg for vreg in range(_TILE_VREGS))


def _tile_row(first: int, tile: int, vreg: int) -> int:
    """Returns the VMEM row one register of one tile occupies within a region."""
    return first + _TILE * tile + _NUM_SUBLANES * vreg


def _emit_row_reduce(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, source: tuple[int, ...],
                     transposed: tuple[int, ...], work: tuple[int, int],
                     combine: Callable[[int, int, int], int], destination: tuple[int, ...],
                     scratch_row: int, turnaround: int) -> int:
    """Reduces each row of one tile across its 16 lanes, leaving the result broadcast back along them.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the reduction starts at.
        source: VREGs holding the tile, register v carrying rows 4v through 4v+3.
        transposed: VREGs the tile transposes into.
        work: The folded partials and the register they are read back into, rewritten in place.
        combine: The operation the rows fold under, being vmax or vadd.
        destination: VREGs the broadcast result lands in, laid out as the source is.
        scratch_row: First VMEM row the fold turns its partials through.
        turnaround: Bundles an access waits after the one whose VMEM rows it reads.

    Returns:
        The next free bundle index.
    """
    folded, shifted = work
    slot = _emit_transpose(put, builder, slot, source=source, destination=transposed)

    # Combines the registers elementwise, every lane now carrying a row, to one partial per sublane.
    put(offset=slot, e=combine(folded, transposed[0], transposed[1]))
    slot += 1
    for register in transposed[2:]:
        put(offset=slot, e=combine(folded, folded, register))
        slot += 1

    # Halves the sublanes through VMEM, whose sublanes read consecutive rows, nothing reducing across them.
    # Leaves stale rows past each shift, so only sublane zero is read on.
    for step, distance in enumerate(_SUBLANE_FOLDS):
        row = scratch_row + 2 * _NUM_SUBLANES * step
        put(offset=slot, x=builder.vst(folded, row))
        slot += 1 + turnaround
        put(offset=slot, x=builder.vld(shifted, row + distance))
        slot += 1
        put(offset=slot, e=combine(folded, folded, shifted))
        slot += 1

    # Spreads sublane zero over consecutive rows, which one load reads back as agreeing sublanes.
    broadcast = scratch_row + 2 * _NUM_SUBLANES * len(_SUBLANE_FOLDS)
    for offset in range(_NUM_SUBLANES):
        put(offset=slot, x=builder.vst(folded, broadcast + offset, mask=_SUBLANE_ZERO))
        slot += 1
    slot += turnaround
    put(offset=slot, x=builder.vld(shifted, broadcast))
    slot += 1
    return _emit_transpose(put, builder, slot, source=(shifted,) * _NUM_SUBLANES, destination=destination)


def _emit_transpose(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, source: tuple[int, ...],
                    destination: tuple[int, ...]) -> int:
    """Stages one tile into the transpose array and drains its columns back out.

    The staging array is neither a VREG nor an SREG, so the schedule holds the distance between the two halves
    rather than the hazard model.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the first vtxin occupies.
        source: VREGs staged in, one per quarter, repeating where one register fills the tile.
        destination: VREGs the columns drain into.

    Returns:
        The next free bundle index.
    """
    for quarter, register in enumerate(source):
        put(offset=slot, e=builder.vtxin(register, quarter))
        slot += _TXIN_II
    slot += _TXIN_TO_TXOUT
    for quarter, register in enumerate(destination):
        put(offset=slot, e=builder.vtxout(register, quarter))
        slot += _TXOUT_II
    return slot


def _emit_tile_update(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, tile: int,
                      registers: FlashRegisterMap, operation: Callable[[int, int, int], int], operand: int,
                      row: int, turnaround: int, store: bool = False) -> int:
    """Applies one elementwise operation to an output tile, reading a spilled tile through the scratch tile.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the update starts at.
        tile: Which output tile is being updated.
        registers: The register map the block is placed under.
        operation: The operation applied, being vmul for a rescale and vadd for a product.
        operand: First VREG of the right operand.
        row: First VMEM row of the output accumulator.
        turnaround: Bundles an access waits after the one whose VMEM rows it reads.
        store: Determines whether a resident tile is written back to VMEM, which the drain needs.

    Returns:
        The next free bundle index.
    """
    if tile < registers.resident:
        base = registers.accumulator + tile * _TILE_VREGS
        for vreg in range(_TILE_VREGS):
            put(offset=slot, e=operation(base + vreg, base + vreg, operand + vreg))
            slot += 1
        if not store:
            return slot
        for vreg in range(_TILE_VREGS):
            put(offset=slot, x=builder.vst(base + vreg, _tile_row(row, tile, vreg)))
            slot += 1
        return slot

    for vreg in range(_TILE_VREGS):
        put(offset=slot, x=builder.vld(registers.scratch + vreg, _tile_row(row, tile, vreg)))
        slot += 1
    for vreg in range(_TILE_VREGS):
        put(offset=slot, e=operation(registers.scratch + vreg, registers.scratch + vreg, operand + vreg))
        slot += 1
    for vreg in range(_TILE_VREGS):
        put(offset=slot, x=builder.vst(registers.scratch + vreg, _tile_row(row, tile, vreg)))
        slot += 1
    return slot + turnaround


def _flash_vmem_map(*, head: int, constants: int = 0, masks: int = 0) -> dict[str, int]:
    """Returns the first VMEM row of each region one attention block stages.

    Args:
        head: Width of the head the block accumulates, which is the rows every operand of it spans.
        constants: Tiles of stated numbers the kernel reads, which ride the queries' own staging.
        masks: Mask tiles each query block stages, one per key block, which follow the constants.

    Returns:
        The first row of each region, keyed by what it stages.
    """
    spans = (("query", head), ("initial", _TILE), ("zero", _TILE), ("constant", constants * _TILE),
             ("mask", masks * _TILE), ("key", head), ("value", head), ("output", head), ("scratch", _TILE * 2),
             ("incoming", head))
    rows, cursor = {}, _ROW_FLASH_BASE
    for name, span in spans:
        rows[name] = cursor
        cursor += span
    return rows


def _emit_flash_product(put: Callable[..., None], builder: asm_module.AsmBuilder, slot: int, *, weight: int,
                        pushed: int, result: int) -> int:
    """Streams one stationary tile through the array and pops the product one token group leaves.

    One token group pushes `_TILE_VREGS` times, which the matrix output FIFO holds without a pop in between, so
    the run is every push and then every pop.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        slot: The bundle index the load occupies.
        weight: First VREG of the stationary tile.
        pushed: First VREG of the tile streamed against it.
        result: First VREG the product pops into.

    Returns:
        The next free bundle index.
    """
    put(offset=slot, m=builder.vmatload(weight))
    slot += _MATLOAD_HOLD
    for vreg in range(_TILE_VREGS):
        put(offset=slot, m=builder.vmatpush(pushed + vreg))
        slot += _PUSH_II
    for vreg in range(_TILE_VREGS):
        put(offset=slot, m=builder.vmatpop(result + vreg))
        slot += _POP_II
    return slot
