"""Provides the register maps and the VMEM allocator that say where an image's values live.
"""

from __future__ import annotations

from dataclasses import dataclass

from bf16 import (_MAX_DESCRIPTOR_ROWS, _TILE, _TILE_VREGS,
                  _VMEM_ENTRIES_PER_LANE)
from isa_model import PROFILE


class LayoutError(ValueError):
    """Reports a layout the machine cannot express, or two that do not compose.

    Raised rather than repaired, because an inserted relayout is a launch and a launch costs about 1.2 ms of fixed time
    however little it moves.
    """


@dataclass(frozen=True)
class RegisterMap:
    """Holds where each of a GEMM step's operands lives in the vector register file.

    Only the accumulators scale with the block's width, since a step stages one weight tile and pops one result tile at
    a time.

    Attributes:
        accumulator: First register of the accumulators, which hold `resident` output tiles.
        activation: First register of the activation tiles, one per token group.
        weight: First register of the weight tile the array is loaded from.
        result: First register a matrix result pops into.
        token_groups: Token groups one block covers, being its rows over the tile.
        output_tiles: Output tiles one block covers, being its columns over the tile.
        resident: Output tiles the accumulators hold in registers. The rest accumulate in VMEM, and zero means every
            one of them does.
        scratch: First register of the tiles a spilled accumulator is read into and written back from, or None when
            every output tile stays in registers and none is needed.
        scratch_tiles: Scratch tiles the map hands out, being one per token group where the file affords it.
    """

    accumulator: int
    activation: int
    weight: int
    result: int
    token_groups: int
    output_tiles: int
    resident: int
    scratch: int | None = None
    scratch_tiles: int = 0

    @property
    def used(self) -> int:
        """Returns the registers the map hands out."""
        if self.scratch is None:
            return self.result + _TILE_VREGS
        return self.scratch + self.scratch_tiles * _TILE_VREGS

    @property
    def spilled(self) -> int:
        """Returns the output tiles that accumulate in VMEM rather than in registers."""
        return self.output_tiles - self.resident

    @property
    def reload_hoisted(self) -> bool:
        """Returns whether every token group owns a scratch tile, which lets its reload issue at a step's head."""
        return bool(self.spilled) and self.scratch_tiles == self.token_groups


@dataclass(frozen=True)
class VmemRegion:
    """Holds the rows of one VMEM slot that one staged tensor owns.

    Attributes:
        name: What asked for it, so a collision report can say who.
        base: First row.
        rows: Rows per feature tile.
        tile_stride: Rows between one feature tile's base and the next. Equal to
            `rows` unless a token owns a rounded-up number of them.
        tiles: Feature tiles staged here.
    """

    name: str
    base: int
    rows: int
    tile_stride: int
    tiles: int

    @property
    def span(self) -> int:
        """Returns the rows this region occupies."""
        return self.tile_stride * (self.tiles - 1) + self.rows

    @property
    def end(self) -> int:
        """Returns one past the region's last row."""
        return self.base + self.span

    def row(self, tile: int = 0, offset: int = 0) -> int:
        """Returns the VMEM row of `offset` within feature tile `tile`."""
        if not 0 <= tile < self.tiles:
            message = (f"Unable to address tile {tile} of region {self.name!r}, "
                       f"which has {self.tiles}.")
            raise LayoutError(message)
        return self.base + tile * self.tile_stride + offset


@dataclass(frozen=True, slots=True)
class FlashRegisterMap:
    """Holds where one attention block's tiles live in the vector register file.

    The running statistics and the accumulator are rewritten by every key block, so they hold dedicated registers
    throughout. Everything else outlasts only the phase that wrote it, so the phases share one transient region.

    Attributes:
        transient: First register of the region the phases share.
        row_max: First register of the running row maxima, held broadcast across each sublane's lanes.
        row_sum: First register of the running row sums, held the same way.
        accumulator: First register of the output accumulators, which hold `resident` output tiles.
        scratch: First register of the tile a spilled accumulator is rescaled through, or None when none spills.
        token_groups: Token groups one block covers, being its queries over the tile.
        output_tiles: Output tiles the head covers, being its width over the tile.
        resident: Output tiles the accumulators hold in registers. The rest accumulate in VMEM.
    """

    transient: int
    row_max: int
    row_sum: int
    accumulator: int
    scratch: int | None
    token_groups: int
    output_tiles: int
    resident: int

    @property
    def per_tile(self) -> int:
        """Returns the registers one tile of every token group occupies."""
        return self.token_groups * _TILE_VREGS

    @property
    def weight(self) -> int:
        """Returns the first register of the stationary operand a vmatload streams from."""
        return self.transient

    @property
    def pushed(self) -> int:
        """Returns the first register of the tile a vmatpush streams into the array."""
        return self.transient + _TILE_VREGS

    @property
    def result(self) -> int:
        """Returns the first register a matrix result pops into."""
        return self.pushed + self.per_tile

    @property
    def partial(self) -> int:
        """Returns the first register the contraction steps of one score tile accumulate into.
        """
        return self.result + self.per_tile

    @property
    def scores(self) -> int:
        """Returns the first register of the score tile, which reuses the matrix phase's registers."""
        return self.transient

    @property
    def staged(self) -> int:
        """Returns the first register of the tiles vtxout transposes a score tile into."""
        return self.scores + self.per_tile

    @property
    def running(self) -> int:
        """Returns the first register of the running reduction partials, one per token group."""
        return self.staged + self.per_tile

    @property
    def transient_span(self) -> int:
        """Returns the registers the shared region spans, being the widest phase's demand."""
        return max(_TILE_VREGS + 3 * self.per_tile, 2 * self.per_tile + self.token_groups)

    @property
    def used(self) -> int:
        """Returns the registers the map hands out."""
        if self.scratch is None:
            return self.accumulator + self.resident * self.per_tile
        return self.scratch + _TILE_VREGS

    @property
    def spilled(self) -> int:
        """Returns the output tiles that accumulate in VMEM rather than in registers."""
        return self.output_tiles - self.resident


class VmemAllocator:
    """Hands out disjoint VMEM regions, so two staged tensors cannot overlap.

    Bump allocation, because a launch's regions all live as long as the launch. A descriptor names its VMEM slot in its
    immediate, so an allocator spanning slots would produce addresses the D slot and the X slot disagree about.
    """

    def __init__(self, *, slot: int = 0,
                 capacity: int = _VMEM_ENTRIES_PER_LANE) -> None:
        self.slot = slot
        self.capacity = capacity
        self._next = 0
        self._regions: list[VmemRegion] = []

    @property
    def regions(self) -> tuple[VmemRegion, ...]:
        """Returns every region handed out, in the order it was asked for."""
        return tuple(self._regions)

    @property
    def used(self) -> int:
        """Returns the rows handed out."""
        return self._next

    def region(self, name: str, rows: int, *, tiles: int = 1) -> VmemRegion:
        """Reserves `tiles` feature tiles of `rows` rows each.

        Args:
            name: What is being staged, for collision and overflow reports.
            rows: Rows one feature tile occupies.
            tiles: Feature tiles staged side by side, `rows` apart. The
                sublanes read consecutive rows, so nothing collides and the
                stride needs no padding.

        Returns:
            The region.

        Raises:
            LayoutError: If it does not fit.
        """
        if rows < 1 or tiles < 1:
            message = (f"Unable to reserve {tiles} tiles of {rows} rows for "
                       f"{name!r}. Both must be positive.")
            raise LayoutError(message)
        if rows > _MAX_DESCRIPTOR_ROWS:
            message = (
                f"Unable to fill {name!r} with one descriptor: {rows} rows past "
                f"the {_MAX_DESCRIPTOR_ROWS} the row field encodes. Split it, or "
                f"widen the field -- DMA_RESERVED_W has 85 spare bits."
            )
            raise LayoutError(message)
        region = VmemRegion(name, self._next, rows, rows, tiles)
        if region.end > self.capacity:
            message = (
                f"Unable to place {name!r}: {region.span} rows at {self._next} "
                f"runs past the slot's {self.capacity}. Already placed: "
                f"{', '.join(f'{r.name}[{r.base}:{r.end}]' for r in self._regions)}"
            )
            raise LayoutError(message)
        self._next = region.end
        self._regions.append(region)
        return region


    def check_disjoint(self) -> None:
        """Raises if any two regions overlap.
        """
        ordered = sorted(self._regions, key=lambda r: r.base)
        for earlier, later in zip(ordered, ordered[1:]):
            if later.base < earlier.end:
                message = (
                    f"VMEM regions overlap: {earlier.name!r} occupies "
                    f"[{earlier.base}:{earlier.end}) and {later.name!r} starts at "
                    f"{later.base}. Nothing in hardware checks this and a launch "
                    f"boundary does not clear VMEM, so the collision is silent."
                )
                raise LayoutError(message)


def register_map(*, block_m: int, block_n: int) -> RegisterMap:
    """Returns where a block's operands live in the register file, and how much of it accumulates in registers.

    The registers a step's own staging leaves over decide how many output tiles stay resident, the rest accumulating in
    VMEM through a scratch tile.

    Args:
        block_m: Rows one block covers, a whole number of the tile.
        block_n: Columns one block covers, a whole number of the tile.

    Returns:
        The map.

    Raises:
        LayoutError: If either extent is not a positive whole number of tiles, if a step's own staging does not fit
            in the register file, or if a block that must spill leaves no room for the scratch tile it reads into.
    """
    for name, extent in (("block_m", block_m), ("block_n", block_n)):
        if extent < _TILE or extent % _TILE:
            message = (
                f"Unable to map a {name} of {extent}. A block covers a whole number of {_TILE} wide tiles, because "
                f"the systolic array is that wide."
            )
            raise LayoutError(message)

    token_groups, output_tiles = block_m // _TILE, block_n // _TILE
    accumulator = 0
    per_tile = token_groups * _TILE_VREGS
    staging = per_tile + _TILE_VREGS + _TILE_VREGS
    entries = PROFILE["vreg_entry_count"]
    if staging > entries:
        message = (
            f"Unable to stage a block of {block_m} rows: its {token_groups} activation tiles, weight tile and "
            f"result tile want {staging} of the {entries} registers present. Narrow the block's rows."
        )
        raise LayoutError(message)
    # A tile the registers cannot hold accumulates in VMEM, which needs somewhere to read it into and add against,
    # so a block that spills any of them buys that scratch out of what it could otherwise keep resident.
    affordable = (entries - staging) // per_tile
    if affordable >= output_tiles:
        resident, scratch_rows, scratch_tiles = output_tiles, 0, 0
    else:
        scratch_rows = _TILE_VREGS
        # The staging check above passes a block whose staging fills the file exactly, which leaves the scratch
        # tile nowhere to sit. A spill needs it, so this is the block's real floor rather than the staging alone.
        if staging + scratch_rows > entries:
            message = (
                f"Unable to spill a block of {block_m} rows by {block_n} columns: its staging takes {staging} of "
                f"the {entries} registers present, which leaves no room for the {scratch_rows} register scratch "
                f"tile a spilled accumulator is read into. Narrow the block's rows."
            )
            raise LayoutError(message)
        # Every tile spills once any does. A spilled tile drains as two partials, one per VMEM bank, and a
        # resident one as a single tile, so a block holding both leaves the drain ragged where the host reads it
        # as one rectangle.
        resident = 0
        # A scratch tile per token group lets each group's reload issue before the group ahead of it has folded,
        # which is what puts the reload under a push instead of in front of its own adds.
        scratch_tiles = min(token_groups, (entries - staging) // _TILE_VREGS)

    activation = accumulator + resident * per_tile
    weight = activation + per_tile
    result = weight + _TILE_VREGS
    return RegisterMap(accumulator=accumulator, activation=activation, weight=weight, result=result,
                       token_groups=token_groups, output_tiles=output_tiles, resident=resident,
                       scratch=result + _TILE_VREGS if scratch_rows else None, scratch_tiles=scratch_tiles)


def flash_register_map(*, queries: int, head: int) -> FlashRegisterMap:
    """Returns where one attention block's tiles sit, refusing an allocation the file cannot hold.

    Notes:
        A key block's width does not enter this budget, its scores staging in VMEM and being read back one tile at a
        time. The head's width is what the accumulator costs, and the accumulator is what spills.

    Args:
        queries: Queries one block covers, a whole number of tiles.
        head: Width of the head the block accumulates, a whole number of tiles.

    Returns:
        The map.

    Raises:
        LayoutError: If either extent is not a whole number of tiles, if the transient region and the running
            statistics do not fit in the register file, or if a spilling block has no room for its scratch tile.
    """
    for name, extent in (("queries", queries), ("head", head)):
        if extent < _TILE or extent % _TILE:
            message = (
                f"Unable to map a {name} of {extent}. A block covers a whole number of {_TILE} wide tiles, "
                f"because the transpose array and the systolic array are both that wide."
            )
            raise LayoutError(message)

    token_groups, output_tiles = queries // _TILE, head // _TILE
    per_tile = token_groups * _TILE_VREGS
    entries = PROFILE["vreg_entry_count"]

    # The phases take turns, so the shared region is the widest of them rather than their sum.
    transient_span = max(_TILE_VREGS + 3 * per_tile, 2 * per_tile + token_groups)
    row_max = transient_span
    row_sum = row_max + per_tile
    accumulator = row_sum + per_tile

    if accumulator > entries:
        message = (
            f"Unable to stage a block of {queries} queries: its shared phase region and running statistics want "
            f"{accumulator} of the {entries} registers present, before the accumulator it carries across the key "
            f"loop. Narrow the block's queries."
        )
        raise LayoutError(message)

    fields = dict(transient=0, row_max=row_max, row_sum=row_sum, accumulator=accumulator,
                  token_groups=token_groups, output_tiles=output_tiles)
    free = entries - accumulator
    if free // per_tile >= output_tiles:
        return FlashRegisterMap(**fields, scratch=None, resident=output_tiles)

    # A tile the registers cannot hold accumulates in VMEM, which needs somewhere to be rescaled against, so a
    # block that spills any of them buys that scratch out of what it could otherwise keep resident.
    if free < _TILE_VREGS:
        message = (
            f"Unable to spill a block of {queries} queries by a head of {head}: its staging takes {accumulator} "
            f"of the {entries} registers present, which leaves no room for the {_TILE_VREGS} register scratch "
            f"tile a spilled accumulator is rescaled through. Narrow the block's queries."
        )
        raise LayoutError(message)

    resident = (free - _TILE_VREGS) // per_tile
    return FlashRegisterMap(**fields, scratch=accumulator + resident * per_tile, resident=resident)
