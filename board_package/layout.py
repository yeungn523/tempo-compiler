"""Where a tensor's elements live, and what that costs.

Three levels, and only two of them are choices.

**DMEM** is an ABI. A tensor written by one launch is read by the next, and no
compiler sees that program -- `run_gpt2_forward.py` is Python calling drivers -- so
the layout is written down rather than inferred. :class:`Tiled` is that writing.

**VMEM** is per-launch scratch, and until now it was fourteen hardcoded row
constants in `bf16.py` with no allocator. A kernel could not ask for "so many rows
at a conflict-free pitch"; it was assigned a region by hand, and two kernels sharing
a constant clobbered each other silently. :class:`VmemAllocator` is the fix, and it
is the thing that makes VMEM tiling programmable rather than hardcoded.

**VREG** is not a choice. A VREG is `NUM_SUBLANES x NUM_LANES` = 64 BF16 and always
holds four VMEM rows; the only freedom is WHICH four, which is the sublane stride,
and which axis they walk. Mosaic's `VectorLayout` carries bitwidth, offsets, tiling
and an implicit dim because a TPU vreg is 8x128 and a tile can sit in it many ways.
Ours has one way. That is why this file is two integers where theirs is six.

Everything else here is derivation. Given a layout and an access, the descriptor
fields, the `vld` fields and the bank-conflict cost follow by arithmetic --
`tools/pilot_layout_derivation.py` checks that against every call site the shipping
kernels use, at three GPT-2 shapes and at Qwen's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from bf16 import (_ELEMENT_BYTES, _MAX_DESCRIPTOR_ROWS, _NUM_LANES,
                  _NUM_SUBLANES, _ROW_BYTES, _TILE, _VMEM_ENTRIES_PER_LANE)
from asm import SUBLANE_STRIDE_W

#: The largest sublane stride the X slot encodes, which caps a padded pitch.
MAX_SUBLANE_STRIDE: int = (1 << SUBLANE_STRIDE_W) - 1

#: Axes a VREG's four sublanes can walk. The same DMEM layout costs differently
#: depending on which one an access uses, and that is the whole banking story:
#: a GEMM walks tokens (stride 1 when packed, conflict-free) while a softmax walks
#: feature tiles (stride `tile_stride`, which is why the pitch gets padded).
TOKEN = "token"
FEATURE_TILE = "feature_tile"


class LayoutError(ValueError):
    """A layout the machine cannot express, or two that do not compose.

    Raised rather than repaired. On this machine an inserted relayout is a launch,
    and a launch costs about 1.2 ms of fixed time no matter how little it moves --
    so silently fixing a mismatch would hide the single most expensive thing a
    kernel can do. The message names both layouts and the cheapest fix.
    """


@dataclass(frozen=True)
class Tiled:
    """A DMEM layout: where element ``(*index, feature)`` of a tensor lives.

        lane = feature mod NUM_LANES
        row  = sum(index[i] * strides[i]) + (feature div NUM_LANES) * tile_stride

    Attributes:
        shape: The LOGICAL shape, whose last axis is features. Rank is not two --
            a stacked per-head operand is rank 3, and the head stride is exactly the
            number that was being computed by hand and getting silently wrong.
        strides: Row stride per leading axis, one shorter than `shape`.
        tile_stride: Rows between one feature tile and the next.
        replicated: The value does not vary along the leading axis, so one row
            stands for all of them. A bias is one row; storing `rows` copies of it
            costs 42.5 MB of DMEM across twelve layers at 256 rows.
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
        """What `pack_activations` writes: ``row = feature_tile * rows + token``."""
        return Tiled((rows, features), (1,), rows)

    @staticmethod
    def weights(depth: int, columns: int) -> "Tiled":
        """What `pack_weights` writes: ``row = column_tile * depth + k``."""
        return Tiled((depth, columns), (1,), depth)

    @staticmethod
    def row_major(rows: int, features: int) -> "Tiled":
        """The host's own order. Never used on the device except by transpose."""
        return Tiled((rows, features), (features // _NUM_LANES,), 1)

    @staticmethod
    def broadcast(rows: int, features: int) -> "Tiled":
        """One row standing for `rows` of them -- a bias, or LayerNorm's gamma."""
        return Tiled((rows, features), (0,), 1, replicated=True)

    @staticmethod
    def stacked(count: int, rows: int, features: int, *, stride: int) -> "Tiled":
        """`count` packed operands end to end, entry h starting `stride` rows on."""
        return Tiled((count, rows, features), (stride, 1), rows)

    # -- what the compiler derives ------------------------------------------

    @property
    def feature_tiles(self) -> int:
        """Feature tiles one entry spans."""
        return self.shape[-1] // _NUM_LANES

    @property
    def token_stride(self) -> int:
        """Rows one step along the token axis moves."""
        return self.strides[-1]

    @property
    def leading_stride(self) -> int:
        """Rows between consecutive entries of the OUTERMOST axis."""
        return self.strides[0]

    def extent_rows(self) -> int:
        """Rows one entry of the outermost axis occupies."""
        return self.feature_tiles * self.tile_stride

    def sublane_stride(self, axis: str) -> int:
        """Rows between consecutive sublanes of a VREG walking `axis`."""
        if axis == TOKEN:
            return self.token_stride
        if axis == FEATURE_TILE:
            return self.tile_stride
        message = f"Unable to walk axis {axis!r}; it is {TOKEN!r} or {FEATURE_TILE!r}."
        raise LayoutError(message)

    def conflict(self, axis: str) -> int:
        """Cycles one VREG access costs: gcd(stride, NUM_SUBLANES).

        `vpu_bank_select` sends sublane s to bank ``(row + s * stride) mod
        NUM_SUBLANES``, so a stride sharing a factor with the bank count collides.
        Reported rather than fixed: which of PACK, DMA, PAD or a functional unit
        answers it is a cost trade only the caller can make.
        """
        return math.gcd(self.sublane_stride(axis), _NUM_SUBLANES)

    def descriptor_stride_bytes(self) -> int:
        """The DMA row stride, in bytes, walking the token axis."""
        return self.token_stride * _ROW_BYTES

    def advance_after(self, inner_walk_rows: int) -> int:
        """Rows to move after the inner loops, on top of what they already walked.

        This is `head_advance`, and deriving it wrong produced two silent
        generality bugs -- both times because two constants GPT-2 makes equal were
        assumed equal everywhere. It is one formula.

        Args:
            inner_walk_rows: Rows the enclosed loops advance on their own.

        Returns:
            The residual advance, which may be zero or negative.
        """
        return self.leading_stride - inner_walk_rows

    def composes_with(self, other: "Tiled") -> bool:
        """Whether a consumer in `other` can read this without a relayout."""
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
                   f"destination row, so re-pitching on the way in is free.")
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


def conflict_free_pitch(rows: int) -> int:
    """The smallest pitch at or above `rows` that is coprime with the bank count.

    The oldest fix in the shared-memory literature -- pad the leading dimension so
    the column stride is coprime with the bank count -- and the one this machine can
    express, because a DMA descriptor names its own destination row. For a
    power-of-two bank count it is simply the next odd number.
    """
    pitch = rows
    while math.gcd(pitch, _NUM_SUBLANES) != 1:
        pitch += 1
    return pitch


@dataclass(frozen=True)
class VmemRegion:
    """Rows of one VMEM slot, owned by one staged tensor.

    Attributes:
        name: What asked for it, so a collision report can say who.
        base: First row.
        rows: Rows per feature tile.
        pitch: Rows between one feature tile's base and the next. Equal to `rows`
            when nothing is padded, and one more when the pitch was made odd.
        tiles: Feature tiles staged here.
    """

    name: str
    base: int
    rows: int
    pitch: int
    tiles: int

    @property
    def span(self) -> int:
        """Rows this region occupies, padding included."""
        return self.pitch * (self.tiles - 1) + self.rows

    @property
    def end(self) -> int:
        """One past the last row."""
        return self.base + self.span

    def row(self, tile: int = 0, offset: int = 0) -> int:
        """The VMEM row of `offset` within feature tile `tile`."""
        if not 0 <= tile < self.tiles:
            message = (f"Unable to address tile {tile} of region {self.name!r}, "
                       f"which has {self.tiles}.")
            raise LayoutError(message)
        return self.base + tile * self.pitch + offset

    def conflict(self) -> int:
        """Cycles a VREG access walking feature tiles costs here."""
        return math.gcd(self.pitch, _NUM_SUBLANES)


class VmemAllocator:
    """Hands out disjoint VMEM regions, at pitches that do not collide.

    This replaces fourteen hardcoded row constants in `bf16.py`. They worked, and
    they are why a kernel could not choose its own tiling: raising the token block
    meant re-deriving the map by hand, two kernels naming the same constant
    clobbered each other silently across a launch boundary, and
    `MAX_ATTENTION_ROWS = 320` is a consequence of where `_ROW_SOFTMAX_OUT` was put
    rather than of how much VMEM there is.

    Bump allocation, because a launch's regions all live as long as the launch.
    Slot selection stays explicit: a DMA descriptor names its VMEM slot in the
    immediate, so an allocator that silently spanned slots would produce addresses
    the D slot and the X slot disagree about.
    """

    def __init__(self, *, slot: int = 0,
                 capacity: int = _VMEM_ENTRIES_PER_LANE) -> None:
        self.slot = slot
        self.capacity = capacity
        self._next = 0
        self._regions: list[VmemRegion] = []

    @property
    def regions(self) -> tuple[VmemRegion, ...]:
        """Everything handed out, in the order it was asked for."""
        return tuple(self._regions)

    @property
    def used(self) -> int:
        """Rows handed out."""
        return self._next

    def region(self, name: str, rows: int, *, tiles: int = 1,
               conflict_free: bool = True) -> VmemRegion:
        """Reserves `tiles` feature tiles of `rows` rows each.

        Args:
            name: What is being staged, for collision and overflow reports.
            rows: Rows one feature tile occupies.
            tiles: Feature tiles staged side by side.
            conflict_free: Pad the pitch so a VREG walking feature tiles does not
                collide. Costs `tiles - 1` rows and nothing else. Pass False when
                the access walks tokens instead, where the pitch is irrelevant.

        Returns:
            The region.

        Raises:
            LayoutError: If it does not fit, or the padded pitch exceeds what a
                sublane stride encodes.
        """
        if rows < 1 or tiles < 1:
            message = (f"Unable to reserve {tiles} tiles of {rows} rows for "
                       f"{name!r}. Both must be positive.")
            raise LayoutError(message)
        pitch = conflict_free_pitch(rows) if conflict_free and tiles > 1 else rows
        if tiles > 1 and pitch > MAX_SUBLANE_STRIDE:
            message = (
                f"Unable to walk a pitch of {pitch} for {name!r}. The sublane "
                f"stride field is {SUBLANE_STRIDE_W} bits, so it reaches "
                f"{MAX_SUBLANE_STRIDE}. Stage fewer rows per tile, or walk the "
                f"token axis instead so the pitch does not enter the access."
            )
            raise LayoutError(message)
        if rows > _MAX_DESCRIPTOR_ROWS:
            message = (
                f"Unable to fill {name!r} with one descriptor: {rows} rows past "
                f"the {_MAX_DESCRIPTOR_ROWS} the row field encodes. Split it, or "
                f"widen the field -- DMA_RESERVED_W has 85 spare bits."
            )
            raise LayoutError(message)
        region = VmemRegion(name, self._next, rows, pitch, tiles)
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
