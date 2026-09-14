"""Provides the launch plan a kernel's matmuls imply, with the padding and repacking they need."""

from __future__ import annotations

import math

from bf16 import _BLOCK_DEPTH, _BLOCK_TOKENS, _TILE, _ZERO_TILE_WORDS
from layout import LayoutError, register_map

from .ir import (_EMITTED_NEST, _REDUCTION_AXIS, Drain, FusedLaunch, Launch, MatMul, Padding, Plan,
                 ProgramError, Repack)

_MINIMUM_LOOP_BLOCKS: int = 2
"""Smallest number of accumulation blocks a group carries, set by the two way unroll of the block loop."""

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

_AXIS_NOUNS: dict[str, str] = {"m": "tokens", "n": "output features", "k": "reduction"}
"""What each axis's extent counts, for the messages that report one."""


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
