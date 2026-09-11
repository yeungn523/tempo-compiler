"""Provides the layouts that say where a tensor's elements live, and what each one costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from bf16 import (_MAX_DESCRIPTOR_ROWS, _NUM_LANES, _ROW_BYTES, _TILE, _TILE_VREGS,
                  _VMEM_ENTRIES_PER_LANE)
from isa_model import PROFILE
from asm import AGU_SHIFT_W


TOKEN = "token"
FEATURE_TILE = "feature_tile"


class LayoutError(ValueError):
    """Reports a layout the machine cannot express, or two that do not compose.

    Raised rather than repaired, because an inserted relayout is a launch and a launch costs about 1.2 ms of fixed
    time however little it moves. The message names both layouts and the cheapest fix.
    """


@dataclass(frozen=True)
class Tiled:
    """Describes a DMEM layout, being where element `(*index, feature)` of a tensor lives.

        lane = feature mod NUM_LANES
        row  = sum(index[i] * strides[i]) + (feature div NUM_LANES) * tile_stride

    Attributes:
        shape: The logical shape, whose last axis is features. A stacked per-head operand is rank three, so the
            rank is not fixed at two.
        strides: Row stride per leading axis, one shorter than `shape`.
        tile_stride: Rows between one feature tile and the next.
        replicated: Determines whether one row stands for every entry of the leading axis, which is what a bias
            is.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    tile_stride: int
    replicated: bool = False

    def __post_init__(self) -> None:
        if len(self.shape) < 2:
            message = (
                f"Unable to lay out a rank-{len(self.shape)} tensor. The last axis "
                f"is features and there must be at least one axis before it."
            )
            raise LayoutError(message)
        if len(self.strides) != len(self.shape) - 1:
            message = (
                f"Unable to lay out shape {self.shape} with {len(self.strides)} "
                f"strides. One per leading axis is {len(self.shape) - 1}."
            )
            raise LayoutError(message)
        if self.shape[-1] % _NUM_LANES:
            message = (
                f"Unable to lay out {self.shape[-1]} features. A row is "
                f"{_NUM_LANES} lanes and a partial row has no encoding."
            )
            raise LayoutError(message)

    # -- the layouts the pass actually uses ---------------------------------

    @staticmethod
    def packed(rows: int, features: int) -> "Tiled":
        """Returns the layout `pack_activations` writes, being `row = feature_tile * rows + token`."""
        return Tiled((rows, features), (1,), rows)

    @staticmethod
    def weights(depth: int, columns: int) -> "Tiled":
        """Returns the layout `pack_weights` writes, being `row = column_tile * depth + k`."""
        return Tiled((depth, columns), (1,), depth)

    @staticmethod
    def row_major(rows: int, features: int) -> "Tiled":
        """Returns the host's own order, which reaches the device only through a transpose."""
        return Tiled((rows, features), (features // _NUM_LANES,), 1)

    @staticmethod
    def token_major(rows: int, features: int) -> "Tiled":
        """Returns the layout the packed softmax stages, being `row = token * stride + tile`.

        The transpose of `packed`, so an access walking feature tiles reads consecutive rows. The stride is
        `rows_per_token` of the tiles a token owns.
        """
        return Tiled((rows, features), (rows_per_token(features // _NUM_LANES),), 1)

    @staticmethod
    def broadcast(rows: int, features: int) -> "Tiled":
        """Returns the layout of one row standing for `rows` of them, which is what a bias is."""
        return Tiled((rows, features), (0,), 1, replicated=True)

    @staticmethod
    def stacked(count: int, rows: int, features: int, *, stride: int) -> "Tiled":
        """Returns `count` packed operands end to end, each starting `stride` rows past the last."""
        return Tiled((count, rows, features), (stride, 1), rows)

    # -- what the compiler derives ------------------------------------------

    @property
    def feature_tiles(self) -> int:
        """Returns the feature tiles one entry spans."""
        return self.shape[-1] // _NUM_LANES

    @property
    def token_stride(self) -> int:
        """Returns the rows one step along the token axis moves."""
        return self.strides[-1]

    @property
    def leading_stride(self) -> int:
        """Returns the rows between consecutive entries of the outermost axis."""
        return self.strides[0]

    def extent_rows(self) -> int:
        """Returns the rows one entry of the outermost axis occupies."""
        return self.feature_tiles * self.tile_stride

    def rows_between_sublanes(self, axis: str) -> int:
        """Returns the rows between consecutive sublanes of a VREG walking `axis`.

        One means the access is readable as it lies: a `vld` reads
        NUM_SUBLANES CONSECUTIVE rows. Anything else has to be closed by the
        layout, by packing DMEM or by landing the rows where the reader walks,
        because the sublanes read consecutive rows and nothing else.
        """
        if axis == TOKEN:
            return self.token_stride
        if axis == FEATURE_TILE:
            return self.tile_stride
        message = f"Unable to walk axis {axis!r}; it is {TOKEN!r} or {FEATURE_TILE!r}."
        raise LayoutError(message)

    def descriptor_stride_bytes(self) -> int:
        """Returns the DMA row stride in bytes, walking the token axis."""
        return self.token_stride * _ROW_BYTES

    def advance_after(self, inner_walk_rows: int) -> int:
        """Returns the rows to move after the inner loops, on top of what they already walked.

        Args:
            inner_walk_rows: Rows the enclosed loops advance on their own.

        Returns:
            The residual advance, which may be zero or negative.
        """
        return self.leading_stride - inner_walk_rows

    def composes_with(self, other: "Tiled") -> bool:
        """Determines whether a consumer in `other` reads this layout without a relayout."""
        return (self.token_stride == other.token_stride
                and self.tile_stride == other.tile_stride)

    def require_composes(self, other: "Tiled", *, producer: str, consumer: str) -> None:
        """Raises unless `other` reads this layout as it lies.

        Raises:
            LayoutError: Naming both layouts and the cheapest fix. A relayout is a
                launch here, so this never inserts one.
        """
        if self.composes_with(other):
            return
        if self.token_stride == other.token_stride:
            fix = (f"change one tile_stride: {self.tile_stride} vs "
                   f"{other.tile_stride}. A DMA descriptor names its own "
                   f"destination row, so restriding on the way in is free.")
        else:
            fix = (f"the token strides differ ({self.token_stride} vs "
                   f"{other.token_stride}), so this needs a scatter -- a launch.")
        message = (
            f"Unable to feed {producer} into {consumer} as it lies. "
            f"{producer} is {self}, {consumer} wants {other}. Cheapest fix: {fix}"
        )
        raise LayoutError(message)

    def __str__(self) -> str:
        tag = ", replicated" if self.replicated else ""
        return (f"Tiled(shape={self.shape}, strides={self.strides}, "
                f"tile_stride={self.tile_stride}{tag})")


@dataclass(frozen=True)
class RegisterMap:
    """Holds where each of a GEMM step's operands lives in the vector register file.

    The regions are laid end to end from register zero, so the registers a tiling leaves free are contiguous and the
    next tiling can claim them. A step stages one output tile's weights and pops one output tile's results at a
    time, so only the accumulators scale with the block's width.

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


class VmemAllocator:
    """Hands out disjoint VMEM regions, so two staged tensors cannot overlap.

    Bump allocation, because a launch's regions all live as long as the launch. Slot selection stays explicit: a
    descriptor names its VMEM slot in the immediate, so an allocator spanning slots would produce addresses the D
    slot and the X slot disagree about.
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

    def token_major_region(self, name: str, tokens: int, *, tiles: int) -> VmemRegion:
        """Reserves `tokens` rows of `tiles` consecutive feature tiles each.

        A token is contiguous here, so an access walking its tiles reads consecutive rows and the tokens are what
        sit a stride apart. That stride is a power of two, because the AGU reaches a token by shifting the loop
        index rather than multiplying it.

        Args:
            name: What is being staged, for collision and overflow reports.
            tokens: Tokens staged.
            tiles: Feature tiles one token owns.

        Returns:
            The region, whose `rows` is one token's tiles and whose
            `tile_stride` is that rounded up to a power of two.

        Raises:
            LayoutError: If it does not fit, or one token needs more rows than a
                descriptor moves.
        """
        if tokens < 1 or tiles < 1:
            message = (f"Unable to reserve {tokens} tokens of {tiles} tiles for "
                       f"{name!r}. Both must be positive.")
            raise LayoutError(message)
        if tiles > _MAX_DESCRIPTOR_ROWS:
            message = (
                f"Unable to gather {tiles} tiles for {name!r} in one descriptor: "
                f"past the {_MAX_DESCRIPTOR_ROWS} the row field encodes."
            )
            raise LayoutError(message)
        region = VmemRegion(name, self._next, tiles, rows_per_token(tiles), tokens)
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

        Bump allocation cannot produce an overlap, so this exists to catch a
        hand-placed region being mixed in with allocated ones during the migration
        off the constants -- which is exactly when a silent clobber would appear.
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

    A step stages one activation set per token group, one weight tile and one result tile, so those are placed first
    and the registers left over decide how many output tiles accumulate in registers. The rest accumulate in VMEM,
    which costs a scratch tile to read them into and one fewer resident with it.

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


def rows_per_token(tiles: int) -> int:
    """Returns the VMEM rows one token owns when its feature tiles land consecutively.

    Rounded up to a power of two, because the AGU reaches a token by shifting the loop index rather than multiplying
    it. The padding never exceeds `tiles` rows a token and costs nothing to walk.

    Args:
        tiles: Feature tiles one token's row covers.

    Returns:
        Rows per token, a power of two at or above `tiles`.

    Raises:
        LayoutError: If the stride needs a shift wider than the AGU's field.
    """
    if tiles < 1:
        message = f"Unable to lay out {tiles} feature tiles. At least one is needed."
        raise LayoutError(message)
    per_token = 1 << (tiles - 1).bit_length()
    max_shift = (1 << AGU_SHIFT_W) - 1
    if per_token.bit_length() - 1 > max_shift:
        message = (
            f"Unable to walk {per_token} rows per token. The AGU shift field is "
            f"{AGU_SHIFT_W} bits, so the shift reaches {max_shift} and the stride "
            f"{1 << max_shift}."
        )
        raise LayoutError(message)
    return per_token
