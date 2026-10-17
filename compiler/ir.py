"""Provides the types a kernel is read into and the launches a plan states."""

from __future__ import annotations

from dataclasses import dataclass, field

from bf16 import _TILE
from layout import register_map

_LANE_AXIS: dict[str, str] = {"activation": "k", "weight": "n", "output": "n"}
"""Dimension each operand reads across the 16 lanes, which is the one its memory must step through by one."""

_REDUCTION_AXIS: str = "k"
"""Axis the contraction walks, which the nest names alongside the output axes."""

_COLUMN_AXIS: str = "n"
"""Axis the output columns walk, whose place in the nest decides who issues the blocks."""

_EMITTED_NEST: tuple[str, ...] = ("m", "n", "k")
"""Order the emitter walks the axes in, being the token blocks a host issues, then the columns, then the
contraction. A launch whose kernel wrote another order is emitted in this one."""


class ProgramError(ValueError):
    """Raised when a kernel is not one this machine can launch."""


@dataclass(frozen=True, slots=True)
class Guard:
    """Describes a predicate this machine runs by padding rather than by predicating.

    Notes:
        The array holds no per element predicate, so a mask is runnable only when the elements it excludes are a
        trailing run the host zero fills going in and slices off coming out.

    Attributes:
        axis: Induction variable the predicate bounds, being "m", "n" or "k".
        bound: Extent the axis is bounded to, as the parameter naming it or an integer.
    """

    axis: str
    bound: int | str


@dataclass(frozen=True, slots=True)
class Access:
    """Describes how one operand is addressed.

    Attributes:
        role: What the operand is to the machine, being "activation", "weight" or "output".
        base: Pointer argument the access offsets from.
        strides: Stride the access advances by per induction variable, as the parameter naming it.
        guards: One guard per axis the access's predicate bounds, empty when it carries no predicate.
    """

    role: str
    base: str | None
    strides: dict[str, str | int | None]
    guards: tuple[Guard, ...]

    @property
    def lane_axis(self) -> str:
        """Returns the dimension this operand is read across the lanes, which its memory must step through by one."""
        return _LANE_AXIS[self.role]


@dataclass(frozen=True, slots=True)
class TileLoop:
    """Describes one loop around a matmul that walks the output tile rather than the contraction.

    Attributes:
        trips: Iterations the loop runs.
        step: Positions along the output axis one iteration advances.
        bound: Position the loop stops at, which is the extent the kernel states.
        axis: Output axis the loop walks, read from the operand its induction variable addresses, or None when
            neither operand's address depends on it.
    """

    trips: int
    step: int
    bound: int
    axis: str | None = None


@dataclass(frozen=True, slots=True)
class Operation:
    """One operation standing between a matmul and the value its consumer reads.

    Lowering a chain needs the order its operations run in and the values each one reads.

    Attributes:
        name: The operation's name, a reduce carrying the combiner it folds under.
        position: Where the operation sits in program order.
        result: Identifier of the value it produces.
        operands: Identifier of each value it reads.
    """

    name: str
    position: int
    result: int
    operands: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Carried:
    """One value a loop hands from each trip to the next.

    A carried value outlasts every trip, so it holds a dedicated register region, while a value one trip
    computes shares the transient one.

    Attributes:
        position: Which of the loop's carried values this is.
        entering: Identifier of the value it enters the loop holding.
        inside: Identifier the body reads it under.
        leaving: Identifier of the value each trip hands on.
        result: Identifier the value carries outside the loop, which is what a later operation reads it under.
        shape: Extent of the value, which tells a running statistic from an accumulator.
        chain: The operations updating it each trip, in program order.
    """

    position: int
    entering: int
    inside: int
    leaving: int
    result: int
    shape: tuple[int, ...]
    chain: tuple[Operation, ...] = ()


@dataclass(frozen=True, slots=True)
class MatMul:
    """Describes one matmul and what a launch needs from it.

    Attributes:
        rows: Rows of the output tile.
        columns: Columns of the output tile.
        depth: Reduction one loop iteration accumulates.
        accumulator: Element type the matmul accumulates in.
        epilogue: Operations standing between this matmul and its consumer.
        prologue: Operations standing between this matmul's left operand and the matmul itself.
        carried: Values the reduction loop hands from each trip to the next, empty where it walks none.
        epilogue_chain: The epilogue's operations in program order, each carrying what it reads.
        prologue_chain: The prologue's operations in program order, each carrying what it reads.
        accumulator_chain: The operations producing what the matmul accumulates into, which is where a
            rescale of the running product lands once Triton folds it onto the dot.
        product: The dot itself, which carries the value it leaves and the values it reads. The chains around a
            matmul terminate at it rather than naming it, so this is what joins them into one graph.
        constants: The number each constant the kernel states carries, paired with the value it defines. A
            constant reaches the vector unit as a staged tile, so the number is what a host writes.
        left_source: Index of the matmul producing the left operand, or None when it comes from memory.
        reduction: Whole reduction the loop walks, or None when its bound is a runtime argument.
        tile_loops: Every loop walking the output rather than the contraction, outermost first, holding the ones
            whose bounds are compile time constants. A kernel walking its tiles on the grid leaves this empty.
        nest: Axis each enclosing loop walks, outermost first, holding "k" for the contraction. An axis whose loop
            ran once is absent, since canonicalization folds it before the reader sees the module.
        accesses: How each operand is addressed, keyed by role.
        grid_axes: Output axes the kernel's grid walks, named as the loops are. An axis the grid carries is
            walked outside the module, so it reaches a launch rather than the module and states no extent the
            plan can read from the loops. This is what tells such an axis from one whose single trip loop
            canonicalization folded, which states the tile it asked for and nothing else.
    """

    rows: int
    columns: int
    depth: int
    accumulator: str
    epilogue: frozenset[str]
    prologue: frozenset[str]
    left_source: int | None
    reduction: int | None
    tile_loops: tuple[TileLoop, ...]
    nest: tuple[str, ...]
    accesses: dict[str, Access]
    grid_axes: frozenset[str] = frozenset()
    carried: tuple[Carried, ...] = ()
    epilogue_chain: tuple[Operation, ...] = ()
    prologue_chain: tuple[Operation, ...] = ()
    accumulator_chain: tuple[Operation, ...] = ()
    product: Operation | None = None
    constants: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Padding:
    """Describes one axis a launch covers by padding rather than exactly.

    Attributes:
        matmul: Position of the matmul whose problem is padded.
        axis: Induction variable the padding grows, being "m", "n" or "k".
        extent: Extent the caller asked for.
        padded: Extent whole launches cover.
    """

    matmul: int
    axis: str
    extent: int
    padded: int


@dataclass(frozen=True, slots=True)
class Repack:
    """Describes one operand the host must rewrite in memory before a launch can read it.

    Attributes:
        matmul: Position of the matmul reading the operand.
        role: What the operand is to the machine.
        base: Pointer argument the operand arrives on.
        axis: Dimension the machine reads across the 16 lanes.
        stride: Elements stepped to move one position along that dimension, which must become one.
    """

    matmul: int
    role: str
    base: str | None
    axis: str
    stride: int | None


@dataclass(frozen=True, slots=True)
class Drain:
    """Describes what one launch writes back, so every host reads it without deriving anything of its own.

    Attributes:
        zero_tile_words: Words of clearing tile the drain buffer opens with, which a reader skips.
        words: Words the drain buffer holds, the clearing tile included.
        shape: Extent of each axis of the drained partials, as blocks by partials by the block's output tiles by
            token rows by lanes, following the loops the image walks. The second axis is the one a host sums
            over, and is one when the device summed the groups itself.
    """

    zero_tile_words: int
    words: int
    shape: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Launch:
    """Holds the arguments the GEMM builder takes for one matmul.

    Attributes:
        row_blocks: Token blocks the product covers, being launches the host issues rather than a trip count,
            since the image holds one block at a time.
        column_tiles: Output column tiles the product covers.
        groups: Reduction groups each column tile accumulates.
        loop_blocks: Accumulation blocks one group carries.
        activate: Determines whether each staged tile is activated entering the array, which is where this
            machine computes the transform a kernel writes ahead of its matmul.
        block_m: Token rows one block covers, which the emitter lays its registers out for.
        block_n: Output columns one block covers, which `column_tiles` cannot say because it counts tiles rather
            than the blocks holding them.
        nest: Axis each loop walks in the order the kernel wrote them, outermost first. The emitter walks
            `_EMITTED_NEST`, so a launch naming another order is emitted in that one.
        partials: Partial products each column tile leaves, being one for a block the registers hold. A block
            accumulating in VMEM leaves two, since consecutive blocks fold into opposite banks.
    """

    row_blocks: int
    column_tiles: int
    groups: int
    loop_blocks: int
    activate: bool
    block_m: int
    block_n: int
    nest: tuple[str, ...] = _EMITTED_NEST
    partials: int = field(init=False)

    def __post_init__(self) -> None:
        """Settles the partial count from the block, which decides where the accumulators live."""
        spilled = register_map(block_m=self.block_m, block_n=self.block_n).spilled
        object.__setattr__(self, "partials", 2 if spilled else 1)

    @property
    def output_tiles(self) -> int:
        """Returns the output tiles one block covers, being its columns over the tile."""
        return self.block_n // _TILE

    @property
    def blocks(self) -> int:
        """Returns the blocks the column loop walks, being the product's tiles over the tiles one block covers."""
        return self.column_tiles // self.output_tiles

    @property
    def issues_columns(self) -> bool:
        """Returns whether a host issues one launch per column block rather than one image walking them all."""
        return self.nest[0] == _COLUMN_AXIS

    @property
    def image_blocks(self) -> int:
        """Returns the column blocks one image walks, being one when a host issues them itself."""
        return 1 if self.issues_columns else self.blocks

    @property
    def launches(self) -> int:
        """Returns the launches a host issues, being a token block each and a column block each when it walks them."""
        return self.row_blocks * (self.blocks if self.issues_columns else 1)


@dataclass(frozen=True, slots=True)
class FusedLaunch:
    """Describes one image running a matmul pair and the transform between them.

    The hidden width is walked one tile at a time, each chunk contracted, activated and consumed where it lies,
    so the intermediate never reaches memory and the pair costs one launch rather than two.

    Attributes:
        row_blocks: Token blocks a host issues this image for.
        chunks: Tiles the hidden width covers, being the loop both matmuls sit inside.
        steps: Tiles the producing matmul's contraction covers.
        output_tiles: Tiles the consuming matmul's output covers, swept inside each chunk.
        produces: Position of the matmul leaving the intermediate.
        consumes: Position of the matmul reading it.
    """

    row_blocks: int
    chunks: int
    steps: int
    output_tiles: int
    produces: int
    consumes: int

    @property
    def launches(self) -> int:
        """Returns the launches a host issues, being one per token block."""
        return self.row_blocks


@dataclass(frozen=True, slots=True)
class FlashLaunch:
    """Describes one image running an attention block's whole key loop.

    The running statistics and the output accumulator hold vector registers for the key loop, so the score
    tile never reaches memory and the whole problem costs one launch however many keys or queries it covers.

    Attributes:
        row_blocks: Query blocks the problem covers, which the image walks itself in one launch.
        key_blocks: Blocks the key loop walks, each covering a tile of keys.
        queries: Queries one block covers.
        head: Width of the head the block accumulates.
        body: The steps one trip of the key loop runs, empty where the caller states the extents alone.
        epilogue: The steps the block runs once the key loop has ended.
        constants: Number of every tile the kernel states, which a host stages ahead of the keys.
        entering: Value the running maximum is staged at, being what the kernel enters its loop holding.
    """

    row_blocks: int
    key_blocks: int
    queries: int
    head: int
    body: tuple = ()
    epilogue: tuple = ()
    constants: tuple[float, ...] = ()
    entering: float = 0.0

    @property
    def output_tiles(self) -> int:
        """Returns the tiles the head covers, being its width over the tile."""
        return self.head // _TILE

    @property
    def launches(self) -> int:
        """Returns the launches a host issues, the image walking every query block itself."""
        return 1


@dataclass(frozen=True, slots=True)
class Pointwise:
    """Describes a kernel computing every output element from the elements its operands hold at the same position.

    Attributes:
        operands: Identifier of each loaded operand in the module, the left one first.
        broadcast_right: Determines whether the right operand is one feature row broadcast down the tokens.
        chain: The operations between the loads and the store, in program order.
        features: Features one token row holds.
    """

    operands: tuple[int, int]
    broadcast_right: bool
    chain: tuple[Operation, ...]
    features: int


@dataclass(frozen=True, slots=True)
class PointwiseLaunch:
    """Holds what one pointwise image is emitted from, the host issuing it once per token block.

    Attributes:
        chunks: Descriptor chunks one token block spans.
        broadcast_right: Determines whether the right operand is one feature row broadcast down the tokens.
        body: The vector operations one VREG runs.
    """

    chunks: int
    broadcast_right: bool
    body: tuple


@dataclass(frozen=True, slots=True)
class RowTerm:
    """Describes one value a row kernel computes, as the operation producing it over the values it reads.

    Attributes:
        name: "input" for the streamed tensor's element, "row" for a per-feature row's, "constant" for a stated
            number, "sum" for the total of its operand over a token's features, or the operation's name otherwise.
        operands: The terms the value is computed from.
        value: The constant's number, or the row's pointer parameter, and None otherwise.
    """

    name: str
    operands: tuple[RowTerm, ...] = ()
    value: float | str | None = None

    @property
    def per_token(self) -> bool:
        """Returns whether the term holds one value per token, which every feature of that token shares."""
        if self.name in ("constant", "sum"):
            return True
        if self.name in ("input", "row"):
            return False
        return all(operand.per_token for operand in self.operands)


@dataclass(frozen=True, slots=True)
class Rows:
    """Describes a kernel computing each token's output row from that token's input row and its sums over it.

    Attributes:
        output: The term the kernel stores, per element.
        rows: Pointer parameter of each per-feature row the output reads, in the order the signature declares them.
        features: Features one token row holds.
    """

    output: RowTerm
    rows: tuple[str, ...]
    features: int


@dataclass(frozen=True, slots=True)
class RowLaunch:
    """Holds what one row image is emitted from, the host issuing it once per token block.

    Attributes:
        features: Features one token row holds.
        rows: Per-feature rows the image stages, in the order the host packs them.
        scalars: Number each broadcast constant the image stages holds, in the order the host packs them.
        prologue: The register loads the image runs once, before its token loop.
        body: The operations one token runs, every register already assigned.
    """

    features: int
    rows: int
    scalars: tuple[float, ...]
    prologue: tuple
    body: tuple


@dataclass(frozen=True, slots=True)
class Plan:
    """Holds the launches a kernel becomes and what the host must stage for them.

    Attributes:
        launches: One launch per matmul, in the order the host issues them, a fused one standing
            for the pair it runs in one image and a flash one for the attention block it runs.
        drains: What each launch writes back, in the order the host issues them.
        padding: Every axis grown to reach whole launches, which the host zero fills and slices back.
        repack: Every operand the host must rewrite in memory, because the lanes read one contiguous run.
        masks_covered: Every predicate the padding stands in for, as the access carrying it and the bound it
            holds its axis under. One the padding cannot reproduce is refused when the kernel is read.
        issued_over_useful: Multiplies the padded shapes issue over the multiplies the problems hold.
    """

    launches: tuple[Launch | FusedLaunch | FlashLaunch, ...]
    drains: tuple[Drain, ...]
    padding: tuple[Padding, ...]
    repack: tuple[Repack, ...]
    masks_covered: tuple[str, ...]
    issued_over_useful: float
