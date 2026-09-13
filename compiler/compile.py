"""Provides the structural reading of a compiled Triton kernel and the launch plan it implies."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asm import AsmKernel
from bf16 import _BLOCK_DEPTH, _BLOCK_TOKENS, _TILE, _ZERO_TILE_WORDS
from emit import Staging, fused_ffn_stream, looped_gemm_stream
from layout import LayoutError, register_map
from triton.runtime import JITFunction

from .frontend import compile_to_ttir

_MINIMUM_LOOP_BLOCKS: int = 2
"""Smallest number of accumulation blocks a group carries, set by the two way unroll of the block loop."""

_ASSUMED_ACCUMULATOR_ELEMENT: str = "f32"
"""Element type a dot accumulates in when its printed result type states none."""

_GELU_OPERATIONS: frozenset[str] = frozenset({"arith.addf", "arith.divf", "arith.mulf", "math.erf"})
"""Operations the exact GELU is written in, being the error function form the SFU's activation implements. The
hyperbolic tangent approximation reads as a different set."""

_EPILOGUES: frozenset[frozenset[str]] = frozenset({frozenset()})
"""Operations the drain runs after a matmul, being none: it writes the product as the array left it. A transform
written after a matmul is a further launch, unless the matmul consuming that product writes it ahead of itself,
where `_PROLOGUES` fuses it into that launch's staging."""

_PROLOGUES: dict[frozenset[str], str] = {
    frozenset(): "",
    _GELU_OPERATIONS: "gelu",
}
"""Transform the staging fuses, keyed by the operations standing between a matmul's left operand and the matmul.
The tile entering the array is where this machine activates, so such a transform rides that tile's staging."""

_LANE_AXIS: dict[str, str] = {"activation": "k", "weight": "n", "output": "n"}
"""Dimension each operand reads across the 16 lanes, which is the one its memory must step through by one."""

_AXIS_VARIABLES: dict[str, tuple[str, str]] = {
    "activation": ("m", "k"),
    "weight": ("k", "n"),
    "output": ("m", "n"),
}
"""Induction variable each tensor axis of each operand carries, before any transpose."""

_REDUCTION_AXIS: str = "k"
"""Axis the contraction walks, which the nest names alongside the output axes."""

_COLUMN_AXIS: str = "n"
"""Axis the output columns walk, whose place in the nest decides who issues the blocks."""

_EMITTED_NEST: tuple[str, ...] = ("m", "n", "k")
"""Order the emitter walks the axes in, being the token blocks a host issues, then the columns, then the
contraction. A launch whose kernel wrote another order is emitted in this one."""

_AXIS_NOUNS: dict[str, str] = {"m": "tokens", "n": "output features", "k": "reduction"}
"""What each axis's extent counts, for the messages that report one."""

_PENDING_AXIS: int = -1
"""Axis a rank one iota occupies until an expand_dims operation places it in a ranked tensor."""

_ADDRESS_OPERATIONS: frozenset[str] = frozenset(
    {"arith.constant", "tt.splat", "tt.broadcast", "tt.make_range", "tt.expand_dims", "tt.addptr", "arith.muli",
     "arith.addi", "tt.get_program_id", "arith.extsi", "arith.trunci"}
)
"""Operations forming an address or supplying a literal rather than a step of an epilogue."""

_PREDICATE_RESHAPES: frozenset[str] = frozenset(
    {"tt.broadcast", "tt.expand_dims", "arith.extsi", "arith.trunci"}
)
"""Operations that carry a predicate into its access's shape without changing which elements it selects."""

_POINTWISE_TRANSFORMS: frozenset[str] = frozenset(
    {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.extf", "arith.truncf", "arith.maxnumf",
     "arith.minnumf", "math.erf", "math.exp", "math.tanh", "math.sqrt", "math.rsqrt"}
)
"""Operations transforming a tile element by element, which change the values it holds and never where it lies. A
walk looking for the memory behind a matmul operand passes through these, leaving the addressing of an operand a
transform stands in front of readable."""

_WIDTH_CASTS: frozenset[str] = frozenset({"arith.extf", "arith.truncf"})
"""Operations changing an element's width alone, which a traced transform records none of. The machine stages one
width whatever width the kernel wrote its arithmetic in, so recording them would key one fused transform under as
many sets as there are ways to spell its precision."""

_ORDERING_PREDICATES: dict[int, tuple[bool, bool]] = {
    2: (True, False), 3: (True, True), 6: (True, False), 7: (True, True),
    4: (False, False), 5: (False, True), 8: (False, False), 9: (False, True),
}
"""Every arith.cmpi predicate that orders its operands, as its enum value to whether it reads "less" and whether it
is inclusive. Signed and unsigned forms both appear because an index is non-negative either way. Which side the axis
falls on is what makes one of these an upper bound, so it is read from the operands rather than assumed."""

_LOAD_LINE: re.Pattern[str] = re.compile(r"=\s*tt\.load\s+(?P<operands>[^:]+):")
"""A printed tt.load, up to the colon that begins its type."""

_CONSTANT_LINE: re.Pattern[str] = re.compile(
    r"^\s*(?P<name>%\S+)\s*=\s*arith\.constant\s+(?:dense<(?P<dense>[^>]*)>|(?P<scalar>[^:]+?))\s*:"
)
"""A printed arith.constant, splat or scalar."""


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
class MatMul:
    """Describes one matmul and what a launch needs from it.

    Attributes:
        rows: Rows of the output tile.
        columns: Columns of the output tile.
        depth: Reduction one loop iteration accumulates.
        accumulator: Element type the matmul accumulates in.
        epilogue: Operations standing between this matmul and its consumer.
        prologue: Operations standing between this matmul's left operand and the matmul itself.
        left_source: Index of the matmul producing the left operand, or None when it comes from memory.
        reduction: Whole reduction the loop walks, or None when its bound is a runtime argument.
        tile_loops: Every loop walking the output rather than the contraction, outermost first, holding the ones
            whose bounds are compile time constants. A kernel walking its tiles on the grid leaves this empty.
        nest: Axis each enclosing loop walks, outermost first, holding "k" for the contraction. An axis whose loop
            ran once is absent, since canonicalization folds it before the reader sees the module.
        accesses: How each operand is addressed, keyed by role.
        stores_inside_reduction: Determines whether the kernel writes its accumulator to memory on every step of
            the contraction, rather than carrying it across the whole of it.
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
    stores_inside_reduction: bool


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
        fuse_groups: Determines whether groups accumulate on the device rather than draining a partial each.
        block_m: Token rows one block covers, which the emitter lays its registers out for.
        block_n: Output columns one block covers, which `column_tiles` cannot say because it counts tiles rather
            than the blocks holding them.
        nest: Axis each loop walks in the order the kernel wrote them, outermost first. The emitter walks
            `_EMITTED_NEST`, so a launch naming another order is emitted in that one.
        partials: Partial products each column tile leaves, being one when the device sums the groups itself. A
            block accumulating in VMEM leaves two of each, since consecutive blocks fold into opposite banks.
    """

    row_blocks: int
    column_tiles: int
    groups: int
    loop_blocks: int
    activate: bool
    fuse_groups: bool
    block_m: int
    block_n: int
    nest: tuple[str, ...] = _EMITTED_NEST
    partials: int = field(init=False)

    def __post_init__(self) -> None:
        """Settles the partial count from the block, which decides where the accumulators live."""
        counted = 1 if self.fuse_groups else self.groups
        spilled = register_map(block_m=self.block_m, block_n=self.block_n).spilled
        object.__setattr__(self, "partials", counted * (2 if spilled else 1))

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
class Plan:
    """Holds the launches a kernel becomes and what the host must stage for them.

    Attributes:
        launches: One launch per matmul, in the order the host issues them, a fused one standing
            for the pair it runs in one image.
        drains: What each launch writes back, in the order the host issues them.
        padding: Every axis grown to reach whole launches, which the host zero fills and slices back.
        repack: Every operand the host must rewrite in memory, because the lanes read one contiguous run.
        masks_covered: Every predicate the padding stands in for, as the access carrying it and the bound it
            holds its axis under. One the padding cannot reproduce is refused when the kernel is read.
        issued_over_useful: Multiplies the padded shapes issue over the multiplies the problems hold.
    """

    launches: tuple[Launch | FusedLaunch, ...]
    drains: tuple[Drain, ...]
    padding: tuple[Padding, ...]
    repack: tuple[Repack, ...]
    masks_covered: tuple[str, ...]
    issued_over_useful: float


def signature_of(kernel: JITFunction, *, operands: dict[str, str]) -> dict[str, str]:
    """Returns the signature a kernel is compiled against, in the order the kernel declares its parameters.

    Args:
        kernel: The function the triton.jit decorator returned.
        operands: Triton type string for each parameter a buffer is passed for, keyed by parameter name.

    Returns:
        The signature.

    Raises:
        ProgramError: If a named operand is not a runtime parameter of the kernel.
    """
    declared = {parameter.name: parameter.is_constexpr for parameter in kernel.params}
    unknown = {name for name in operands if declared.get(name, True)}
    if unknown:
        message = (
            f"Unable to type {', '.join(sorted(unknown))} as an operand of {kernel.__name__}. Every named operand "
            f"must be one of its runtime parameters, which are "
            f"{', '.join(name for name, constexpr in declared.items() if not constexpr)}."
        )
        raise ProgramError(message)
    return {name: "constexpr" if constexpr else operands.get(name, "i32")
            for name, constexpr in declared.items()}


def read_program(kernel: JITFunction, *, signature: dict[str, str],
                 constexprs: dict[str, int]) -> tuple[MatMul, ...]:
    """Reads every matmul a kernel computes, in program order.

    Args:
        kernel: The function the triton.jit decorator returned.
        signature: Parameter name to Triton type string, in declaration order.
        constexprs: Values of the constexpr parameters, which fix the tiles.

    Returns:
        The matmuls.

    Raises:
        ProgramError: If the grid carries independent products, a matmul sits outside every loop, or its loops
            leave the reduction ambiguous.
    """
    graph = _Graph(compile_to_ttir(kernel, signature=signature, constexprs=constexprs), signature=signature)
    axes = {operation.get_int_attr("axis") for operation in graph.by_name("tt.get_program_id")}
    if axes - {0, 1}:
        message = (
            f"Unable to read a grid using axes {sorted(axes)}. Two axes carry the output tile and a third carries "
            f"independent products, whose launch needs the distance the activation pointer moves between them. "
            f"Recovering that distance needs the operand's strides, which this reader does not follow."
        )
        raise ProgramError(message)

    dots = graph.by_name("tt.dot")
    stores = graph.by_name("tt.store")
    consumers = [*dots[1:], *stores]
    accumulators = graph.accumulator_elements()

    matmuls = []
    for index, dot in enumerate(dots):
        loops = graph.enclosing_loops(dot)
        if not loops:
            message = (
                f"Unable to read matmul {index} from no enclosing loop. A dot walks its reduction in a loop, "
                f"because the group that loop accumulates is what the builder's block count comes from."
            )
            raise ProgramError(message)

        rows, columns = dot.get_result(0).get_shape()
        steps = [graph.constant(loop.get_operand(2)) for loop in loops]
        if any(step is None for step in steps):
            message = f"Unable to read the step of matmul {index}'s loops. Each must be a compile time constant."
            raise ProgramError(message)

        pointers = {role: (None if operation is None else operation.get_operand(0))
                    for role, (operation, _) in (("activation", graph._load_behind(dot.get_operand(0))),
                                                 ("weight", graph._load_behind(dot.get_operand(1))))}
        source, prologue = graph.trace(dot.get_operand(0), among=dots)
        at = _reduction_position(index=index, graph=graph, loops=loops, steps=steps, pointers=pointers,
                                 rows=rows, columns=columns, source=source)
        consumer = consumers[index] if index < len(consumers) else None
        epilogue = frozenset() if consumer is None else graph.trace(
            consumer.get_operand(1 if consumer.get_name() == "tt.store" else 0), among=dots)[1]

        def walked(among) -> tuple[TileLoop, ...]:
            """Reads each of the given loops as a tile loop, dropping any whose bounds are runtime values."""
            read = (graph.tile_loop(other, axis=graph.axis_of_loop(other, pointers=pointers)) for other in among)
            return tuple(loop for loop in read if loop is not None)

        matmuls.append(MatMul(
            rows=rows, columns=columns, depth=steps[at],
            accumulator=accumulators[index] if index < len(accumulators) else _ASSUMED_ACCUMULATOR_ELEMENT,
            epilogue=epilogue,
            prologue=prologue,
            left_source=dots.index(source) if source is not None else None,
            reduction=graph.constant(loops[at].get_operand(1)),
            tile_loops=walked([*loops[:at], *loops[at + 1:]]),
            nest=tuple(_REDUCTION_AXIS if position == at else graph.axis_of_loop(loop, pointers=pointers)
                       for position, loop in enumerate(loops)
                       if position == at or graph.axis_of_loop(loop, pointers=pointers) is not None),
            accesses=graph.accesses_of(dot=dot, store=stores[index] if index < len(stores) else None),
            # A store the reduction encloses sits deeper than the reduction itself, which is the
            # accumulator the kernel keeps in memory rather than in registers.
            stores_inside_reduction=(index < len(stores)
                                     and len(graph.enclosing_loops(stores[index])) > at),
        ))
    return tuple(matmuls)


def plan(matmuls: tuple[MatMul, ...], *, strides: dict[str, int],
         dims: tuple[dict[str, int], ...] = (), fuse: bool = True) -> Plan:
    """Plans the launches this machine runs for each matmul, with the padding and repacking they need.

    Notes:
        A problem extent is data the host lays out, so an axis leaving a partial block is padded. A tile is what
        the kernel asked for, so one the array does not compute is refused instead.

    Args:
        matmuls: The matmuls a kernel computes, in program order.
        strides: Value of each stride parameter at the launch, which is where an operand's layout becomes a
            number the machine can judge.
        dims: Extent of each matmul's problem, keyed "m", "n" and "k" for whichever the module leaves open,
            which is an axis put on the grid or a reduction bounded by a runtime argument.
        fuse: Determines whether a matmul pair one image can run is planned as one launch. Clearing it plans
            every matmul on its own, which is the same pair passing its intermediate through memory.

    Returns:
        The plan.

    Raises:
        ProgramError: If a tile, a prologue or an epilogue is one this machine cannot run, a stride is unbound,
            an axis is stated by nothing or by two sources that disagree, or a chained matmul reads a width
            its producer does not leave.
    """
    dims = dims or ({},) * len(matmuls)
    if len(dims) != len(matmuls):
        message = f"Unable to plan {len(matmuls)} matmuls against {len(dims)} problem shapes."
        raise ProgramError(message)
    _check_strides_bound(matmuls, strides=strides)

    # Reads the pairs before planning, a consumer reading registers being one no plain launch describes. A
    # kernel stating one matmul leaves this empty, so its plan is the loop below and nothing else.
    pairs = _fusable_pairs(matmuls) if fuse else {}
    absorbed = _epilogues_absorbed(matmuls)

    launches: list[Launch | FusedLaunch] = []
    drains: list[Drain] = []
    padding: list[Padding] = []
    repack: list[Repack] = []
    masks_covered: list[str] = []
    covered_by_index: dict[int, dict[str, int]] = {}
    planned_at: dict[int, int] = {}
    ratio = 1.0
    consumed = set(pairs.values())
    for index, (matmul, extent) in enumerate(zip(matmuls, dims, strict=True)):
        if index in consumed:
            # Planned with the matmul it is fused onto, whose image already covers this one's contraction.
            continue
        if matmul.epilogue not in _EPILOGUES and {matmul.epilogue} != absorbed.get(index, set()):
            message = (
                f"Unable to fuse an epilogue running {', '.join(sorted(matmul.epilogue))} after matmul "
                f"{index}. The drain writes the product as the array left it, so an epilogue is a further launch "
                f"reading it where it lies. A GELU reaches the array only where the matmul consuming this "
                f"product writes it ahead of itself, which that launch's staging fuses."
            )
            raise ProgramError(message)

        if matmul.prologue not in _PROLOGUES:
            message = (
                f"Unable to fuse a transform running {', '.join(sorted(matmul.prologue))} ahead of matmul "
                f"{index}. The staging fuses a GELU activation and nothing else, so any other transform is a "
                f"further launch writing the tile this one reads."
            )
            raise ProgramError(message)

        blocks = _machine_tile(index=index, matmul=matmul)
        repack.extend(_repacking_needed(index=index, matmul=matmul, strides=strides))
        walked = _tiles_walked(index=index, matmul=matmul)
        stated: dict[str, dict[str, int]] = {}
        for axis, value in (("k", matmul.reduction), *walked.items()):
            if value is not None:
                stated.setdefault(axis, {})["its own loop"] = value
        for axis, value in extent.items():
            stated.setdefault(axis, {})["its caller"] = value
        for role, access in matmul.accesses.items():
            for position, guard in enumerate(access.guards):
                masks_covered.append(f"matmul {index} {role} holds {guard.axis} under {guard.bound}")
                if isinstance(guard.bound, int):
                    # Names each conjunct apart, so two bounds on one axis reach `_stated_extent` as two
                    # sources rather than one.
                    name = f"its {role} mask" if position == 0 else f"its {role} mask ({position + 1})"
                    stated.setdefault(guard.axis, {})[name] = guard.bound
                elif guard.axis not in extent:
                    message = (
                        f"Unable to read the {guard.axis} extent matmul {index} covers from a {role} mask bounded "
                        f"by {guard.bound}, which is a runtime argument. The padding that stands in for the mask "
                        f"is laid out before the launch, so its extent has to be known here: declare the bound a "
                        f"tl.constexpr, or state that axis at the plan."
                    )
                    raise ProgramError(message)

        covered, asked = {}, {}
        # Pads the columns to a whole block rather than a whole tile, since the emitter cannot place a partial
        # block.
        for axis, unit in (("m", matmul.rows), ("n", matmul.columns), ("k", blocks * _BLOCK_DEPTH)):
            asked[axis] = _stated_extent(index=index, axis=axis, unit=unit, sources=stated.get(axis, {}))
            covered[axis] = unit * -(-asked[axis] // unit)
            if covered[axis] != asked[axis]:
                padding.append(Padding(matmul=index, axis=axis, extent=asked[axis], padded=covered[axis]))
            ratio *= covered[axis] / asked[axis]
        if matmul.left_source is not None:
            _check_dimensions_match(index=index, matmul=matmul, covered=covered,
                                    produced=covered_by_index[matmul.left_source])
        covered_by_index[index] = covered
        live = _resident_blocks(index=index, matmul=matmul, covered=covered)
        block_m, block_n = live["m"] * matmul.rows, live["n"] * matmul.columns

        launch = Launch(
            row_blocks=covered["m"] // block_m,
            column_tiles=covered["n"] // _TILE,
            groups=covered["k"] // (blocks * _BLOCK_DEPTH),
            loop_blocks=blocks,
            activate=_PROLOGUES[matmul.prologue] == "gelu",
            # Places the accumulator where the kernel put it, summing the groups on the device for one it
            # carries across the reduction and draining a partial each for one it writes to memory per step.
            fuse_groups=not matmul.stores_inside_reduction,
            # Covers every block the nest keeps live, which is what a tile loop inside the reduction asks for
            # and the kernel's own tile otherwise.
            block_m=block_m,
            block_n=block_n,
            # Places an axis whose loop was folded outermost, where a single trip leaves the order it walks
            # indistinguishable from any other.
            nest=(*(axis for axis in _EMITTED_NEST if axis not in matmul.nest), *matmul.nest),
        )
        planned_at[index] = len(launches)
        launches.append(launch)
        # Measures the drain in the block the launch states, since a narrower one drains fewer rows.
        shape = (launch.image_blocks, launch.partials, launch.output_tiles, launch.block_m, _TILE)
        drains.append(Drain(
            zero_tile_words=_ZERO_TILE_WORDS,
            words=_ZERO_TILE_WORDS + math.prod(shape),
            shape=shape,
        ))
    for produces, consumes in pairs.items():
        _check_fused_width(produces=produces, consumes=consumes, matmuls=matmuls,
                           covered=covered_by_index[produces])
        position = planned_at[produces]
        launches[position] = FusedLaunch(
            row_blocks=covered_by_index[produces]["m"] // matmuls[produces].rows,
            chunks=covered_by_index[produces]["n"] // _TILE,
            steps=covered_by_index[produces]["k"] // _TILE,
            output_tiles=_output_tiles_of(matmuls[consumes]),
            produces=produces, consumes=consumes,
        )
        shape = (1, 1, launches[position].output_tiles, _BLOCK_TOKENS, _TILE)
        drains[position] = Drain(zero_tile_words=_ZERO_TILE_WORDS,
                                 words=_ZERO_TILE_WORDS + math.prod(shape), shape=shape)

    return Plan(launches=tuple(launches), drains=tuple(drains), padding=tuple(padding), repack=tuple(repack),
                masks_covered=tuple(masks_covered), issued_over_useful=ratio)


def stage(left: NDArray[np.float32], right: NDArray[np.float32], *, matmul: MatMul,
          launch: Launch) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Pads an operand pair to the extent its launch covers.

    Args:
        left: The activation operand at the extent the caller holds.
        right: The weight operand at the extent the caller holds.
        matmul: The matmul the kernel computes, whose tile gives the covered extents.
        launch: The planned launch, whose block and trip counts give the covered extents.

    Returns:
        The staged pair.
    """
    rows = launch.row_blocks * launch.block_m
    depth = launch.groups * matmul.depth
    columns = launch.column_tiles * _TILE
    staged = []
    for operand, shape in ((left, (rows, depth)), (right, (depth, columns))):
        # Slices before measuring, so an operand larger than its launch covers is cut rather than broadcast.
        kept = operand[: shape[0], : shape[1]]
        padded = np.zeros(shape, dtype=operand.dtype)
        padded[: kept.shape[0], : kept.shape[1]] = kept
        staged.append(padded)
    return staged[0], staged[1]


def compile_launch(launch: Launch | FusedLaunch) -> tuple[bytes, Staging]:
    """Returns the image one launch compiles to and what a host stages for it.

    Args:
        launch: The launch the planner produced.

    Returns:
        The image bytes padded to the loader's boundary, and the staging the emitter placed them from.

    Raises:
        NotImplementedError: If a plain launch activates its staged tiles, which the emitter does not place.
    """
    if isinstance(launch, FusedLaunch):
        bundles, staging = fused_ffn_stream(chunks=launch.chunks, steps=launch.steps,
                                            output_tiles=launch.output_tiles)
        return AsmKernel(name="fused_ffn", bundles=bundles).to_bytes(pad_to=4), staging

    if launch.activate:
        message = (
            "Unable to compile a launch that activates its staged tiles. The emitter places the staging, the "
            "matrix commands and the drain only, so such a launch comes from `kernels.build_looped_gemm` until "
            "the activation is placed here too."
        )
        raise NotImplementedError(message)
    counts = dict(column_tiles=launch.image_blocks * launch.output_tiles, groups=launch.groups,
                  loop_blocks=launch.loop_blocks,
                  fuse_groups=launch.fuse_groups, block_m=launch.block_m, block_n=launch.block_n)
    bundles, staging = looped_gemm_stream(**counts)
    return AsmKernel(name="looped_gemm", bundles=bundles).to_bytes(pad_to=4), staging


@dataclass(slots=True)
class _Expression:
    """Describes a resolved pointer or index expression.

    Attributes:
        base: Pointer argument the expression offsets from, or None when it carries none.
        axes: Scalar stride multiplying each tensor axis's iota, keyed by axis.
        scalar: The uniform part, as the parameter naming it or an integer.
    """

    base: str | None = None
    axes: dict[int, object] = field(default_factory=dict)
    scalar: object = None


class _Graph:
    """Indexes a compiled module once, then answers every structural question by lookup.

    Args:
        module: The compiled module.
        signature: The signature the module was built from, which names its arguments.

    Attributes:
        module: The module this graph indexes.
        _resolved: Memoized expression per resolved value.
        _zero_filled: Cached zero fill flag per load, read from the printed module on first use.
        _definitions: Defining operation per value.
        _named: Operations grouped by name, in walk order.
        _loop_of_block: Loop owning each body block.
        _body_of_loop: Body block each loop owns.
        _yield_of_loop: Terminator of each loop body.
        _carried: Value each loop body argument entered the loop holding.
        _resolvers: Expression reader per operation name.
        _arguments: Parameter name per entry function argument.
    """

    def __init__(self, module, *, signature: dict[str, str]) -> None:
        self.module = module
        self._resolved: dict[int, _Expression] = {}
        self._zero_filled: list[bool] | None = None
        operations: list = []
        module.walk(operations.append)

        self._definitions: dict[int, object] = {}
        self._named: dict[str, list] = {}
        blocks: dict[int, object] = {}
        for operation in operations:
            self._named.setdefault(operation.get_name(), []).append(operation)
            for index in range(operation.get_num_results()):
                self._definitions[operation.get_result(index).id()] = operation
            block = operation.get_block()
            if block is not None:
                blocks.setdefault(block.get_parent().id(), block)

        self._loop_of_block: dict[int, object] = {}
        self._body_of_loop: dict[int, object] = {}
        self._yield_of_loop: dict[int, object] = {}
        self._carried: dict[int, object] = {}
        for loop in self.by_name("scf.for"):
            body = blocks.get(loop.get_region(0).id())
            if body is None:
                continue
            self._loop_of_block[body.id()] = loop
            self._body_of_loop[id(loop)] = body
            for index in range(1, body.get_num_arguments()):
                # Reads argument zero as the induction variable, and argument index as loop operand two plus
                # index, which is the value it entered the loop holding.
                self._carried[body.get_argument(index).id()] = loop.get_operand(2 + index)
            self._yield_of_loop[id(loop)] = next(
                (operation for operation in self.by_name("scf.yield")
                 if operation.get_block() is not None and operation.get_block().id() == body.id()), None)

        self._resolvers = {
            "tt.broadcast": self._resolve_through, "arith.extsi": self._resolve_through,
            "arith.trunci": self._resolve_through, "tt.splat": self._resolve_splat,
            "arith.constant": self._resolve_constant, "tt.make_range": self._resolve_range,
            "tt.expand_dims": self._resolve_expand_dims, "arith.muli": self._resolve_multiply,
            "arith.addi": self._resolve_add, "tt.addptr": self._resolve_add,
        }

        names = [name for name, kind in signature.items() if kind != "constexpr"]
        function = module.get_function(module.get_entry_func_name())
        self._arguments: dict[int, str] = {
            function.args(index).id(): names[index] for index in range(function.get_num_args())
        }

    def by_name(self, name: str) -> list:
        """Returns every operation of the given name, in walk order."""
        return self._named.get(name, [])

    def enclosing_loops(self, operation) -> list:
        """Returns every loop whose body contains the given operation, outermost first."""
        chain = []
        current = operation
        while current is not None:
            block = current.get_block()
            if block is None or block.id() not in self._loop_of_block:
                break
            current = self._loop_of_block[block.id()]
            chain.append(current)
        return chain[::-1]

    def tile_loop(self, loop, *, axis: str | None = None) -> TileLoop | None:
        """Returns the trips, step and axis of a loop, or None when any of its three bounds is a runtime value."""
        lower, upper, step = (self.constant(loop.get_operand(index)) for index in range(3))
        if lower is None or upper is None or step is None or step <= 0 or upper <= lower:
            return None
        return TileLoop(trips=-(-(upper - lower) // step), step=step, bound=upper, axis=axis)

    def constant(self, value) -> int | None:
        """Returns the value of a constant, or None when it is computed at runtime."""
        defining = self._definitions.get(value.id())
        if defining is None or defining.get_name() != "arith.constant":
            return None
        try:
            return defining.get_constant_value()
        except (RuntimeError, ValueError, TypeError):
            return None

    def addresses_with(self, pointer, *, variable) -> bool:
        """Returns whether one pointer expression depends on the given induction variable.

        A loop walks the axis of whichever operand its variable addresses, which two loops sharing a step still
        answer differently.

        Args:
            pointer: The pointer value to walk back from.
            variable: The loop's induction variable.

        Returns:
            True when the variable is reached.
        """
        wanted = variable.id()
        seen: set[int] = set()
        pending = [pointer]
        while pending:
            current = pending.pop()
            key = current.id()
            if key == wanted:
                return True
            if key in seen:
                continue
            seen.add(key)
            if key in self._carried:
                pending.append(self._carried[key])
            operation = self._definitions.get(key)
            if operation is not None:
                pending.extend(operation.get_operand(index) for index in range(operation.get_num_operands()))
        return False

    def axis_of_loop(self, loop, *, pointers: dict[str, object]) -> str | None:
        """Returns the output axis one loop walks, read from which operand's address its variable reaches.

        Args:
            loop: The loop to name.
            pointers: The activation's and the weight's pointer values, keyed by role.

        Returns:
            "m" when the variable addresses the activation, "n" when it addresses the weight, or None when it
            reaches neither or both.
        """
        body = self._body_of_loop.get(id(loop))
        if body is None or not body.get_num_arguments():
            return None
        variable = body.get_argument(0)
        reached = {role for role, pointer in pointers.items()
                   if pointer is not None and self.addresses_with(pointer, variable=variable)}
        if reached == {"activation"}:
            return "m"
        if reached == {"weight"}:
            return "n"
        return None

    def trace(self, value, *, among: list) -> tuple[object | None, frozenset[str]]:
        """Traces a value back to whichever of the given operations produced it.

        Args:
            value: The value to trace.
            among: The operations that terminate the trace, being the kernel's matmuls.

        Returns:
            The producing operation, or None when the value comes from memory, and the arithmetic operation names
            passed through on the way.
        """
        found: set[str] = set()
        seen: set[int] = set()
        pending = [value]
        source = None

        while pending:
            current = pending.pop()
            key = current.id()
            if key in seen:
                continue
            seen.add(key)
            operation = self._definitions.get(key)
            if operation is None:
                continue

            if operation in among:
                source = operation
                continue
            name = operation.get_name()
            if name == "scf.for":
                terminator = self._yield_of_loop.get(id(operation))
                if terminator is not None:
                    index = next(position for position in range(operation.get_num_results())
                                 if operation.get_result(position).id() == key)
                    pending.append(terminator.get_operand(index))
                continue
            if name == "tt.load" or name in _ADDRESS_OPERATIONS or operation.get_num_operands() == 0:
                continue

            if name not in _WIDTH_CASTS:
                found.add(name)
            pending.extend(operation.get_operand(index) for index in range(operation.get_num_operands()))
        return source, frozenset(found)

    def accesses_of(self, *, dot, store) -> dict[str, Access]:
        """Reads how each operand of one matmul is addressed.

        Notes:
            A transpose between a load and the matmul reverses which variable each loaded axis carries.

        Args:
            dot: The matmul operation.
            store: The store draining it, or None when a later matmul consumes it instead.

        Returns:
            The accesses, keyed by role.
        """
        sources = [(role, self._load_behind(dot.get_operand(index)))
                   for index, role in enumerate(("activation", "weight"))]
        if store is not None:
            sources.append(("output", (store, False)))

        accesses: dict[str, Access] = {}
        for role, (operation, transposed) in sources:
            if operation is None:
                accesses[role] = Access(role=role, base=None, strides={}, guards=())
                continue
            resolved = self.resolve(operation.get_operand(0))
            variables = _AXIS_VARIABLES[role][::-1] if transposed else _AXIS_VARIABLES[role]
            accesses[role] = Access(
                role=role,
                base=resolved.base,
                strides={variables[axis]: stride for axis, stride in resolved.axes.items()
                         if 0 <= axis < len(variables)},
                guards=self.guards_of(operation, role=role, variables=variables),
            )
        return accesses

    def guards_of(self, operation, *, role: str, variables: tuple[str, ...]) -> tuple[Guard, ...]:
        """Reads the predicate on one access as the axes it bounds, refusing one the machine cannot reproduce.

        Args:
            operation: The load or store carrying the predicate.
            role: What the operand is to the machine.
            variables: Induction variable each tensor axis of this operand carries, transpose already applied.

        Returns:
            One guard per bounded axis, empty when the access carries no predicate.

        Raises:
            ProgramError: If the predicate is not a conjunction of axes bounded from above by uniform values, or
                if a masked load reads its excluded elements as anything but zero.
        """
        index = 2 if role == "output" else 1
        if operation.get_num_operands() <= index:
            return ()

        def refuse(reason: str) -> None:
            """Raises with what was read and why padding cannot stand in for it."""
            message = (
                f"Unable to run the predicate on this kernel's {role}, which {reason}. The array has no per "
                f"element predicate, so the only mask a launch reproduces is one bounding each axis from above "
                f"by a uniform value: the host zero fills the trailing run that leaves and slices it off "
                f"coming out. Bound each axis from above, or compute the masked result in a further launch."
            )
            raise ProgramError(message)

        # Skips a store, which names no fill because the launch writes the whole padded tile. A load names one,
        # and the padding writes zeros, so a load asking for anything else computes something else.
        if role != "output" and not self.reads_zero_outside(operation):
            refuse("fills its excluded elements with something other than zero, which the padding cannot write")

        return tuple(self._guard_from(term, placements, variables=variables, refuse=refuse)
                     for term, placements in self._conjuncts(operation.get_operand(index)))

    def reads_zero_outside(self, load) -> bool:
        """Returns True when a masked load fills its excluded elements with zero.

        Notes:
            The MLIR binding returns None for a dense float constant, so every load's fill is read from the
            module's printed form and matched by position. A load with no fill selects every element and reports
            True, and a fill this reader cannot resolve reports False rather than being assumed zero.
        """
        if self._zero_filled is None:
            zeros: set[str] = set()
            flags: list[bool] = []
            for line in self.module.str_nodebug().splitlines():
                constant = _CONSTANT_LINE.match(line)
                if constant is not None:
                    text = (constant.group("dense") or constant.group("scalar") or "").strip()
                    try:
                        if float(text) == 0.0:
                            zeros.add(constant.group("name"))
                    except ValueError:
                        pass
                found = _LOAD_LINE.search(line)
                if found is not None:
                    operands = [operand.strip() for operand in found.group("operands").split(",")]
                    flags.append(len(operands) < 3 or operands[2] in zeros)
            self._zero_filled = flags
        position = self.by_name("tt.load").index(load)
        return self._zero_filled[position] if position < len(self._zero_filled) else False

    def resolve(self, value) -> _Expression:
        """Reads a value as a base pointer plus a per axis affine offset, memoized."""
        key = value.id()
        if key in self._resolved:
            return self._resolved[key]

        if key in self._arguments:
            resolved = _Expression(base=self._arguments[key], scalar=self._arguments[key])
        elif key in self._carried:
            resolved = self.resolve(self._carried[key])
        else:
            operation = self._definitions.get(key)
            handler = self._resolvers.get(operation.get_name()) if operation is not None else None
            resolved = handler(operation) if handler is not None else _Expression()

        self._resolved[key] = resolved
        return resolved

    def accumulator_elements(self) -> tuple[str, ...]:
        """Returns the element type each matmul accumulates in, in program order.

        Notes:
            The MLIR type binding exposes only fp16 and integer predicates, so the dot result types are read from
            the module's printed form.
        """
        elements = []
        for line in self.module.str_nodebug().splitlines():
            if "tt.dot" in line:
                found = re.search(r"->\s*tensor<[0-9x]+x(\w+)>", line)
                elements.append(found.group(1) if found else _ASSUMED_ACCUMULATOR_ELEMENT)
        return tuple(elements)

    def _conjuncts(self, value) -> list:
        """Returns every comparison a predicate is the conjunction of, with what placed each one in the tile.

        Notes:
            A mask is compared at rank one and only then expanded into the operand's shape, so the expand_dims
            axes crossed on the way down to it are carried out with it.

        Args:
            value: The predicate operand of a load or store.

        Returns:
            One comparison and placement pair per term, the comparison being None for a term that is not one and
            the placements running outermost first.
        """
        found, pending = [], [(value, ())]
        while pending:
            current, placements = pending.pop()
            operation = self._definitions.get(current.id())
            if operation is None:
                found.append((None, placements))
                continue
            name = operation.get_name()
            if name == "tt.expand_dims":
                placements = (*placements, operation.get_int_attr("axis"))
                pending.append((operation.get_operand(0), placements))
            elif name in _PREDICATE_RESHAPES:
                # Traces through broadcast and widen, which reshape without moving which axis an element varies along.
                pending.append((operation.get_operand(0), placements))
            elif name == "arith.andi":
                pending.extend((operation.get_operand(side), placements) for side in range(2))
            else:
                found.append((operation, placements))
        return found

    def _guard_from(self, comparison, placements: tuple[int, ...], *, variables: tuple[str, ...],
                    refuse: Callable[[str], None]) -> Guard:
        """Reads one term of a predicate as the axis it bounds and the value it bounds it to.

        Args:
            comparison: The comparison forming this term, or None when the term is not one.
            placements: The expand_dims axes crossed on the way down to the comparison, outermost first.
            variables: Induction variable each tensor axis of this operand carries.
            refuse: Raises with the reason padding cannot stand in for the predicate.

        Returns:
            The guard this term states.
        """
        if comparison is None or comparison.get_name() != "arith.cmpi":
            found = "none" if comparison is None else comparison.get_name()
            refuse(f"joins a term built by {found} rather than an integer comparison")

        ordering = _ORDERING_PREDICATES.get(comparison.get_int_attr("predicate"))
        if ordering is None:
            refuse("compares an axis for equality, which excludes elements that are not one run")
        less, inclusive = ordering

        sides = [self.resolve(comparison.get_operand(side)) for side in range(2)]
        counts = [len(side.axes) for side in sides]
        if sorted(counts) != [0, 1]:
            refuse(f"compares {counts[0]} axes against {counts[1]}, rather than one against none")
        iota_left = counts[0] == 1
        walked, uniform = sides if iota_left else sides[::-1]
        # Reads an axis left of a "less" and right of a "greater" as the same upper bound. The other two
        # orientations bound from below, which excludes a leading run the padding does not hold.
        if less != iota_left:
            refuse("bounds an axis from below, so the elements it excludes lead rather than trail")

        axis, stride = next(iter(walked.axes.items()))
        if stride != 1:
            refuse(f"walks its axis by {stride} rather than one element at a time")
        # Applies the expand_dims innermost first, as `_resolve_expand_dims` does for an address, since the
        # comparison was made at rank one.
        for inserted in reversed(placements):
            axis = (1 - inserted) if axis == _PENDING_AXIS else (axis if axis < inserted else axis + 1)
        if not 0 <= axis < len(variables):
            refuse(f"bounds tensor axis {axis}, which is not one of the {len(variables)} this operand carries")
        if uniform.scalar is None:
            refuse("compares against a value this reader cannot resolve to a parameter or a constant")

        bound = uniform.scalar
        if inclusive:
            if not isinstance(bound, int):
                refuse(f"bounds {variables[axis]} inclusively by {bound}, whose successor is not known here")
            bound += 1
        return Guard(axis=variables[axis], bound=bound)

    def _load_behind(self, value) -> tuple[object | None, bool]:
        """Returns the load producing a matmul operand, and whether a transpose sits in between.

        Args:
            value: The matmul operand to walk back from.

        Returns:
            The load, or None where the paths meet at no load or at more than one, and whether a transpose was
            crossed reaching it.
        """
        reached: dict[tuple[int, bool], tuple[object, bool]] = {}
        seen: set[tuple[int, bool]] = set()
        pending = [(value, False)]

        while pending:
            current, transposed = pending.pop()
            key = (current.id(), transposed)
            if key in seen:
                continue
            seen.add(key)
            operation = self._definitions.get(current.id())
            if operation is None:
                continue

            name = operation.get_name()
            if name == "tt.load":
                reached[(id(operation), transposed)] = (operation, transposed)
                continue
            if name == "tt.trans":
                pending.append((operation.get_operand(0), not transposed))
                continue
            if name in _POINTWISE_TRANSFORMS:
                pending.extend((operation.get_operand(index), transposed)
                               for index in range(operation.get_num_operands()))

        if len(reached) != 1:
            return None, False
        return next(iter(reached.values()))

    def _resolve_through(self, operation) -> _Expression:
        """Reads an operation that reshapes or widens without changing the expression."""
        return self.resolve(operation.get_operand(0))

    def _resolve_splat(self, operation) -> _Expression:
        """Reads a scalar broadcast, which carries the uniform part and no iota."""
        inner = self.resolve(operation.get_operand(0))
        return _Expression(base=inner.base, scalar=inner.scalar)

    def _resolve_constant(self, operation) -> _Expression:
        """Reads a literal."""
        return _Expression(scalar=self.constant(operation.get_result(0)))

    @staticmethod
    def _resolve_range(operation) -> _Expression:  # noqa: ARG004
        """Reads an iota as a unit stride waiting for an expand_dims to place it."""
        return _Expression(axes={_PENDING_AXIS: 1})

    def _resolve_expand_dims(self, operation) -> _Expression:
        """Places a pending iota on the tensor axis an expand_dims operation leaves varying."""
        inner = self.resolve(operation.get_operand(0))
        inserted = operation.get_int_attr("axis")

        def placed(axis: int) -> int:
            """Returns the tensor axis an inner axis moves to once the new one is inserted."""
            return (1 - inserted) if axis == _PENDING_AXIS else (axis if axis < inserted else axis + 1)

        return _Expression(base=inner.base, scalar=inner.scalar,
                           axes={placed(axis): stride for axis, stride in inner.axes.items()})

    def _resolve_multiply(self, operation) -> _Expression:
        """Scales the strides of the varying side of a multiply by the uniform side."""
        left = self.resolve(operation.get_operand(0))
        right = self.resolve(operation.get_operand(1))
        varying, uniform = (left, right) if left.axes else (right, left)
        if varying.axes and uniform.axes:
            return _Expression(scalar=None)
        return _Expression(
            base=varying.base or uniform.base,
            axes={axis: uniform.scalar if stride == 1 else None for axis, stride in varying.axes.items()},
        )

    def _resolve_add(self, operation) -> _Expression:
        """Merges the axes and uniform parts of both sides of an addition."""
        left = self.resolve(operation.get_operand(0))
        right = self.resolve(operation.get_operand(1))
        return _Expression(
            base=left.base or right.base,
            axes={**left.axes, **right.axes},
            scalar=left.scalar if right.scalar is None else right.scalar,
        )


def _reduction_position(*, index: int, graph: _Graph, loops: list, steps: list[int],
                        pointers: dict[str, object], rows: int, columns: int, source=None) -> int:
    """Returns the position of the loop walking one matmul's contraction.

    Notes:
        A tile loop's variable addresses one operand alone, so the contraction is the loop naming none, and two
        the addresses leave unnamed are told apart by their step.

        A matmul reading another's product has no addressable left operand. Its producer fills one step of the
        contraction and so runs inside it, which names the innermost loop enclosing both.

    Args:
        index: Position of the matmul in the kernel.
        graph: The indexed module the loops belong to.
        loops: The loops enclosing the matmul, outermost first.
        steps: The step each of those loops takes.
        pointers: The activation's and the weight's pointer values, keyed by role.
        rows: Token rows the matmul's tile covers.
        columns: Output columns the matmul's tile covers.
        source: The matmul producing this one's left operand, or None when it comes from memory.

    Returns:
        The position of the reduction loop.

    Raises:
        ProgramError: If every loop walks an output axis, if two of them walk neither, or if a matmul reading
            another's product shares no loop with it.
    """
    if source is not None:
        producing = graph.enclosing_loops(source)
        shared = [position for position, loop in enumerate(loops) if loop in producing]
        if shared:
            return shared[-1]
        message = (
            f"Unable to read matmul {index}'s reduction, which it shares no loop with the matmul producing its "
            f"left operand. That product is one step of this contraction, so the matmul leaving it runs inside "
            f"the loop this one contracts over. A matmul reading a whole product at once contracts in a single "
            f"step, which is the reduction loop folded away rather than a loop this reader can name."
        )
        raise ProgramError(message)

    walking = [position for position, loop in enumerate(loops)
               if graph.axis_of_loop(loop, pointers=pointers) is None]
    if len(walking) > 1:
        # Holds a tile loop advancing its operand from the body it encloses, whose own variable reaches neither
        # pointer, to the step its tile gives it.
        walking = [position for position in walking if steps[position] not in (rows, columns)]
    if len(walking) == 1:
        return walking[0]

    if not walking:
        message = (
            f"Unable to read matmul {index}'s reduction, whose {len(loops)} enclosing loops each address one "
            f"operand alone and so walk an output axis. A reduction loop that runs once is folded before this "
            f"reader sees it, so give it more than one trip, or state its depth as a runtime argument."
        )
        raise ProgramError(message)
    message = (
        f"Unable to tell which of matmul {index}'s loops walks the reduction: {len(walking)} of them address "
        f"neither operand alone and step by neither the tile's {rows} rows nor its {columns} columns "
        f"({', '.join(str(steps[position]) for position in walking)}). One loop may walk the contraction, because "
        f"a reader that picked one of these would accumulate the wrong axis."
    )
    raise ProgramError(message)


def _check_dimensions_match(*, index: int, matmul: MatMul, covered: dict[str, int],
                            produced: dict[str, int]) -> None:
    """Refuses a matmul reading a product at other dimensions than the matmul leaving it covers.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul reading the product.
        covered: Padded extent this matmul covers along each axis.
        produced: Padded extent the producing matmul covers along each axis.

    Raises:
        ProgramError: If the token rows or the width the two matmuls meet at disagree.
    """
    for drained, staged, noun in (("m", "m", "token rows"), ("n", "k", "width")):
        if produced[drained] == covered[staged]:
            continue
        message = (
            f"Unable to stage matmul {index}'s left operand from matmul {matmul.left_source}'s product, whose "
            f"{noun} the two state apart: the product covers {produced[drained]} and this matmul reads "
            f"{covered[staged]}. A hidden width is the producing matmul's columns and the consuming matmul's "
            f"reduction at once, so both statements of it have to agree, after the padding each is covered to."
        )
        raise ProgramError(message)


def _fusable_pairs(matmuls: tuple[MatMul, ...]) -> dict[int, int]:
    """Returns the consuming matmul of each one whose product it reads out of the registers that hold it.

    A pair runs in one image only at the shape its registers afford: one tile of hidden width per chunk, one
    output tile per token group, and the activation the staging fuses standing between the two.

    Args:
        matmuls: The matmuls a kernel computes, in program order.

    Returns:
        The position of the consuming matmul, keyed by the position of the one it reads.
    """
    pairs: dict[int, int] = {}
    for consumes, matmul in enumerate(matmuls):
        produces = matmul.left_source
        if produces is None or produces in pairs:
            continue
        producer = matmuls[produces]
        fits = (_PROLOGUES.get(matmul.prologue) == "gelu"
                and matmul.depth == _TILE and not matmul.columns % _TILE
                and producer.columns == _TILE and not producer.prologue and not matmul.epilogue
                and producer.rows == matmul.rows == _BLOCK_TOKENS)
        if fits:
            pairs[produces] = consumes
    return pairs


def _output_tiles_of(matmul: MatMul) -> int:
    """Returns the tiles a fused consumer's output covers, being what its sweep loops over.

    The width is the consumer's own block together with every loop walking its output columns, because the image
    sweeps the whole output inside each chunk rather than holding it live. Reading the block alone would cover
    one iteration of that loop and leave the rest of the output uncomputed, and a kernel has to write the sweep
    as a loop wherever the width is not a power of two, which a block on its own cannot be.

    Args:
        matmul: The matmul reading the intermediate.

    Returns:
        The tiles the sweep covers.
    """
    swept = math.prod(loop.trips for loop in matmul.tile_loops if loop.axis == "n")
    return swept * matmul.columns // _TILE


def _check_fused_width(*, produces: int, consumes: int, matmuls: tuple[MatMul, ...],
                       covered: dict[str, int]) -> None:
    """Refuses a fused pair whose consumer contracts a width its producer does not leave.

    Args:
        produces: Position of the matmul leaving the intermediate.
        consumes: Position of the matmul reading it.
        matmuls: The matmuls a kernel computes.
        covered: Padded extent the producing matmul covers along each axis.

    Raises:
        ProgramError: If the two state the hidden width apart.
    """
    reduction = matmuls[consumes].reduction
    if reduction is not None and reduction != covered["n"]:
        message = (
            f"Unable to fuse matmul {consumes} onto matmul {produces}, whose hidden width the two state apart: "
            f"the product covers {covered['n']} columns and this matmul contracts over {reduction}. One image "
            f"holds the chunk in registers, so the width it walks is the width the producer leaves."
        )
        raise ProgramError(message)


def _epilogues_absorbed(matmuls: tuple[MatMul, ...]) -> dict[int, set[frozenset[str]]]:
    """Returns the prologue each matmul's consumers write, keyed by the position of the matmul they consume.

    A transform between two matmuls stands after the first and ahead of the second, so the reader records it on
    both and the consuming launch's staging runs it. Two consumers writing different prologues leave two entries
    under one key, which no single fused transform covers.

    Args:
        matmuls: The matmuls a kernel computes, in program order.

    Returns:
        The distinct prologues written ahead of each consumed matmul's product, empty for one nothing consumes.
    """
    absorbed: dict[int, set[frozenset[str]]] = {}
    for matmul in matmuls:
        if matmul.left_source is not None:
            absorbed.setdefault(matmul.left_source, set()).add(matmul.prologue)
    return absorbed


def _machine_tile(*, index: int, matmul: MatMul) -> int:
    """Returns the blocks one reduction group carries, refusing a tile the array does not compute.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to judge.

    Returns:
        The accumulation blocks one group carries.

    Raises:
        ProgramError: If any tile extent is one the array does not compute.
    """
    quantum = _BLOCK_DEPTH * _MINIMUM_LOOP_BLOCKS
    if matmul.rows < _TILE or matmul.rows % _TILE:
        message = (
            f"Unable to compute a BLOCK_M of {matmul.rows} in matmul {index}. It must be a whole number of "
            f"{_TILE}, because a token group is as deep as the systolic array."
        )
        raise ProgramError(message)
    # Leaves both the tile width and the register budget to the map, so a block that is not whole tiles and one
    # that does not fit are refused in the same place.
    try:
        register_map(block_m=matmul.rows, block_n=matmul.columns)
    except LayoutError as error:
        message = f"Unable to place matmul {index}'s block of {matmul.rows} by {matmul.columns}. {error}"
        raise ProgramError(message) from error

    if matmul.depth % quantum:
        message = (
            f"Unable to accumulate a reduction step of {matmul.depth} in matmul {index}. It must be a multiple of "
            f"{quantum}, because double buffering alternates the VMEM slot every {_BLOCK_DEPTH} deep block, so a "
            f"group carries an even count of them."
        )
        raise ProgramError(message)
    return matmul.depth // _BLOCK_DEPTH


def _check_strides_bound(matmuls: tuple[MatMul, ...], *, strides: dict[str, int]) -> None:
    """Refuses a launch that has not said what every stride the kernel names is worth.

    Notes:
        An unbound stride reaches `_repacking_needed` as its own name, which is never 1, so the host would
        rewrite data that was already right.

    Args:
        matmuls: The matmuls a kernel computes.
        strides: Value of each stride parameter at the launch.

    Raises:
        ProgramError: If any stride the kernel names is not bound.
    """
    named: dict[str, list[str]] = {}
    for index, matmul in enumerate(matmuls):
        for role, access in matmul.accesses.items():
            for axis, stride in access.strides.items():
                if isinstance(stride, str) and stride not in strides:
                    named.setdefault(stride, []).append(f"matmul {index}'s {role} along {axis}")
    if not named:
        return

    where = ", ".join(f"{stride} addresses {' and '.join(uses)}" for stride, uses in sorted(named.items()))
    message = (
        f"Unable to plan without a value for {', '.join(sorted(named))}. The kernel names the strides and the "
        f"launch says what they are worth, and both halves are needed to tell whether an operand already lies as "
        f"the 16 lanes read it: {where}. Bind each one at the plan, because an unbound stride would report a "
        f"repack from its own name rather than from the layout."
    )
    raise ProgramError(message)


def _stated_extent(*, index: int, axis: str, unit: int, sources: dict[str, int]) -> int:
    """Returns what one axis covers, refusing a kernel whose loops, masks and caller do not agree on it.

    Notes:
        A launch issues whole tiles, so two sources agree when they round up to the same number of them and the
        tightest is what the padding is measured from. Sources rounding up differently name different launches.

    Args:
        index: Position of the matmul in the kernel.
        axis: The axis being read.
        unit: Positions along the axis one whole tile covers.
        sources: What each source says the axis covers, keyed by how it is named in a message.

    Returns:
        The extent, being the tightest source.

    Raises:
        ProgramError: If nothing states the axis, or if two sources name different launches.
    """
    noun = _AXIS_NOUNS[axis]
    if not sources:
        message = (
            f"Unable to plan matmul {index} without the {noun} it covers. Nothing states it: its module carries "
            f"no loop walking {axis} and no mask bounding it, and its caller asked for no extent either. A loop "
            f"leaves nothing behind when it runs once and canonicalization folds it, and an axis put on the grid "
            f"reaches a launch rather than the module, so state it at the plan."
        )
        raise ProgramError(message)

    tiles = {name: -(-value // unit) for name, value in sources.items()}
    if len(set(tiles.values())) > 1:
        disagreement = ", ".join(f"{name} says {sources[name]} ({tiles[name]} of them)"
                                 for name in sorted(sources, key=sources.get))
        message = (
            f"Unable to plan matmul {index}'s {noun}, which are stated more than once and not the same twice: "
            f"{disagreement}. Every source has to cover the same whole tiles of {unit}, because the launch "
            f"issues tiles and nothing finer, and a reader that picked one of these would run a problem the "
            f"kernel does not state."
        )
        raise ProgramError(message)
    return min(sources.values())


def _tiles_walked(*, index: int, matmul: MatMul) -> dict[str, int]:
    """Returns the extent each of a matmul's tile loops covers, keyed by the axis it walks.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to read.

    Returns:
        The extent stated along "m", along "n", or both, holding only the axes a loop states. This is the loop's
        own bound rather than the whole tiles it walks, so the plan pads it.

    Raises:
        ProgramError: If a loop addresses neither operand, or if two loops walk one axis.
    """
    walked: dict[str, int] = {}
    for loop in matmul.tile_loops:
        if loop.axis is None:
            message = (
                f"Unable to read the output axis matmul {index} covers from a loop stepping by {loop.step}. Its "
                f"induction variable addresses neither the activation nor the weight alone, and which operand a "
                f"loop addresses is what names the axis it walks."
            )
            raise ProgramError(message)
        if loop.axis in walked:
            message = (
                f"Unable to read the {loop.axis} extent matmul {index} covers from two loops walking it. One loop "
                f"may walk each output axis, because a reader that cannot tell two apart would report one of them "
                f"as the whole extent."
            )
            raise ProgramError(message)
        walked[loop.axis] = loop.bound
    return walked


def _resident_blocks(*, index: int, matmul: MatMul, covered: dict[str, int]) -> dict[str, int]:
    """Returns the blocks each output axis keeps live inside the reduction, refusing one the array cannot stage.

    Notes:
        A tile loop inside the reduction keeps every block it walks live until the contraction ends, so one
        block covers them all along that axis. Columns past what the register file affords accumulate in VMEM,
        while every token group needs staging registers of its own, which makes the rows the tighter of the two.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to judge.
        covered: Extent the launch covers along each axis, which is where a tile loop bounded by a runtime
            argument gets its trip count from.

    Returns:
        The blocks held live along "m" and along "n", each one where that axis walks outside the reduction.

    Raises:
        ProgramError: If the register file cannot stage the block those live sets imply.
    """
    # Reads the axes from the nest rather than from the steps, which two loops sharing a step leave ambiguous.
    live = {"m": 1, "n": 1}
    for axis in matmul.nest[matmul.nest.index(_REDUCTION_AXIS) + 1:]:
        if axis in live:
            live[axis] = covered[axis] // (matmul.rows if axis == "m" else matmul.columns)

    block_m, block_n = live["m"] * matmul.rows, live["n"] * matmul.columns
    try:
        register_map(block_m=block_m, block_n=block_n)
    except LayoutError as error:
        message = (
            f"Unable to stage matmul {index}'s block of {block_m} rows by {block_n} columns, which is what a "
            f"nest walking {', '.join(matmul.nest)} keeps live. {error} Walk those tiles outside the reduction "
            f"instead, because per accumulator that is the same additions in the same order and it leaves one "
            f"block live however many the product covers."
        )
        raise ProgramError(message) from error
    return live


def _repacking_needed(*, index: int, matmul: MatMul, strides: dict[str, int]) -> list[Repack]:
    """Reports every operand the host must rewrite in memory before this matmul can read it.

    Notes:
        The 16 lanes read one contiguous run, so whichever dimension lands on them must step by one element.
        One that does not is rewritten on the host first, at the cost of a pass over the data.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to judge.
        strides: Value of each stride parameter at the launch.

    Returns:
        The repacks, empty when every operand already lies as the lanes read it.
    """
    found = []
    for role, access in matmul.accesses.items():
        stride = access.strides.get(access.lane_axis)
        bound = strides.get(stride, stride) if isinstance(stride, str) else stride
        if bound != 1:
            found.append(Repack(matmul=index, role=role, base=access.base, axis=access.lane_axis, stride=bound))
    return found
