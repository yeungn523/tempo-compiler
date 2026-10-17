"""Provides the launch plan a kernel's matmuls imply, with the padding and repacking they need."""

from __future__ import annotations

import math

from bf16 import (_BLOCK_DEPTH, _BLOCK_TOKENS, _MAX_DESCRIPTOR_ROWS, _NUM_LANES, _TILE, _VREG_ELEMENTS,
                  _ZERO_TILE_WORDS, staged_maximum)
from layout import LayoutError, flash_register_map, register_map

from .ir import (_EMITTED_NEST, _REDUCTION_AXIS, Drain, FlashLaunch, FusedLaunch, Launch, MatMul, Padding,
                 Plan, Pointwise, PointwiseLaunch, ProgramError, Repack, RowLaunch, Rows)
from .allocation import allocate
from .dataflow import attention_trip
from .lowering import (attention_block, attention_body, attention_epilogue, attention_extents, pointwise_body,
                       row_body, unnameable)

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
        the kernel asked for, so one finer than the array computes is padded up to it, and one taller than the
        register file places is run as several machine tiles the launch walks. An extent nothing states is
        refused rather than assumed, since an axis the grid walks asks for one program's share of it and covers
        more.

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
    # Reads the attention pairs before planning, an online softmax being a chain no plain launch describes.
    attention = {matmuls[index].left_source: index for index in range(len(matmuls))
                 if fuse and attention_block(matmuls, index=index)}

    launches: list[Launch | FusedLaunch] = []
    drains: list[Drain] = []
    padding: list[Padding] = []
    repack: list[Repack] = []
    masks_covered: list[str] = []
    covered_by_index: dict[int, dict[str, int]] = {}
    planned_at: dict[int, int] = {}
    ratio = 1.0
    consumed = set(pairs.values()) | set(attention.values())
    for index, (matmul, extent) in enumerate(zip(matmuls, dims, strict=True)):
        if index in attention:
            launch = _attention_launch(matmuls, produces=index, consumes=attention[index])
            planned_at[index] = len(launches)
            launches.append(launch)
            shape = (launch.row_blocks, 1, launch.output_tiles, launch.queries, _TILE)
            drains.append(Drain(zero_tile_words=0, words=math.prod(shape), shape=shape))
            continue
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

        blocks, machine_rows, machine_columns = _machine_tile(index=index, matmul=matmul)
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
        units = (("m", machine_rows, matmul.rows),
                 ("n", machine_columns, matmul.columns),
                 ("k", blocks * _BLOCK_DEPTH, matmul.depth))
        for axis, unit, kernel_unit in units:
            asked[axis] = _stated_extent(index=index, axis=axis, unit=unit, sources=stated.get(axis, {}),
                                         kernel_unit=kernel_unit, grid_axes=matmul.grid_axes)
            covered[axis] = unit * -(-asked[axis] // unit)
            if covered[axis] != asked[axis]:
                padding.append(Padding(matmul=index, axis=axis, extent=asked[axis], padded=covered[axis]))
            ratio *= covered[axis] / asked[axis]
            if kernel_unit < unit:
                padding.append(Padding(matmul=index, axis=f"{axis}_tile", extent=kernel_unit, padded=unit))
        if matmul.left_source is not None:
            _check_dimensions_match(index=index, matmul=matmul, covered=covered,
                                    produced=covered_by_index[matmul.left_source])
        covered_by_index[index] = covered
        live = _resident_blocks(index=index, matmul=matmul, covered=covered,
                                machine_rows=machine_rows, machine_columns=machine_columns)
        block_m, block_n = live["m"] * machine_rows, live["n"] * machine_columns

        launch = Launch(
            row_blocks=covered["m"] // block_m,
            column_tiles=covered["n"] // _TILE,
            groups=covered["k"] // (blocks * _BLOCK_DEPTH),
            loop_blocks=blocks,
            activate=_PROLOGUES[matmul.prologue] == "gelu",
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


def plan_pointwise(pointwise: Pointwise) -> PointwiseLaunch:
    """Plans the image a pointwise kernel runs, which a host issues once per token block.

    Args:
        pointwise: The kernel the reader recovered.

    Returns:
        The launch.

    Raises:
        ProgramError: If one token block of the row is not a whole number of descriptor chunks, or an operation has
            no vector operation to run as.
    """
    rows = _BLOCK_TOKENS * pointwise.features // _NUM_LANES
    if pointwise.features % _TILE or rows % _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to plan a pointwise image over {pointwise.features} features. One token block of them is {rows} "
            f"VMEM rows, which must be a whole number of the {_MAX_DESCRIPTOR_ROWS} rows one descriptor moves."
        )
        raise ProgramError(message)
    return PointwiseLaunch(chunks=rows // _MAX_DESCRIPTOR_ROWS, broadcast_right=pointwise.broadcast_right,
                           body=pointwise_body(pointwise))


def plan_rows(rows: Rows) -> RowLaunch:
    """Plans the image a row kernel runs, which a host issues once per token block.

    Args:
        rows: The kernel the reader recovered.

    Returns:
        The launch.

    Raises:
        ProgramError: If a row is not a whole number of VREGs or outgrows one descriptor, or the row's arithmetic
            has no vector operation to run as or outnumbers the registers.
    """
    if rows.features % _VREG_ELEMENTS or rows.features // _NUM_LANES > _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to plan a row image over {rows.features} features. `vredsum` reduces one VREG, so a row must be "
            f"a whole number of {_VREG_ELEMENTS} features, and one descriptor gathers at most "
            f"{_MAX_DESCRIPTOR_ROWS * _NUM_LANES} of them."
        )
        raise ProgramError(message)
    scalars, prologue, body = row_body(rows)
    return RowLaunch(features=rows.features, rows=len(rows.rows), scalars=scalars, prologue=prologue, body=body)


def _attention_launch(matmuls: tuple[MatMul, ...], *, produces: int, consumes: int) -> FlashLaunch:
    """Returns the launch one attention block runs, refusing a chain the machine has no vector operation for.

    Args:
        matmuls: The matmuls a kernel computes, in program order.
        produces: Position of the matmul leaving the scores.
        consumes: Position of the matmul contracting the probabilities.

    Returns:
        The launch, carrying the steps its trip was placed into.

    Raises:
        ProgramError: If the online softmax runs an operation the machine cannot name as a step, or if the trip
            admits no order the register file holds its values under.
    """
    for position, chain in ((consumes, matmuls[consumes].prologue_chain),
                            (consumes, matmuls[consumes].epilogue_chain)):
        refused = unnameable(chain)
        if refused:
            message = (
                f"Unable to run matmul {position}'s chain, which names {', '.join(refused)}. Every operation "
                f"standing around an attention block's products has to be one the vector unit runs in a single "
                f"pass, so write those in terms of the operations it carries."
            )
            raise ProgramError(message)

    extents = attention_extents(matmuls, produces=produces, consumes=consumes)
    trip = attention_trip(matmuls, produces=produces, consumes=consumes)
    registers = flash_register_map(queries=extents["queries"], head=extents["head"])
    placement = allocate(trip, registers=registers)
    tiles = registers.output_tiles
    entering = trip.values[trip.carried["row_max"][0]].number
    return FlashLaunch(**extents,
                       body=attention_body(placement, trip=trip, steps=extents["head"] // _TILE, tiles=tiles),
                       epilogue=attention_epilogue(trip, tiles=tiles), constants=trip.constants,
                       entering=staged_maximum(0.0 if entering is None else entering), mask=trip.mask)


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


def _machine_tile(*, index: int, matmul: MatMul) -> tuple[int, int, int]:
    """Returns the blocks one reduction group carries and the tile the array runs the matmul as.

    Notes:
        The array computes tiles of `_TILE`, and double buffering alternates the VMEM slot every `_BLOCK_DEPTH`
        deep block, so a group carries an even count of them. An extent finer than one of those is rounded up
        and padded rather than refused, which costs nothing in accuracy: `stage` zero fills the pad, a zero deep
        block adds nothing to the accumulator, and a padded row or column lands beside the product rather than
        inside it.

        A tile taller than the register file stages is run as several machine tiles instead, the launch walking
        them to cover what the kernel asked for. The columns spill to VMEM, so the file bounds the rows alone
        and a wide tile stands as the kernel wrote it, the image it places being what bounds it. That is refused
        where the image overflows IRAM, which `compile_launch` measures rather than this guesses at.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to judge.

    Returns:
        The accumulation blocks one group carries, the rows one machine tile covers, and its columns.

    Raises:
        ProgramError: If no machine tile places the kernel's rows, or if running the rows as several would move
            an epilogue that reduces across the tile.
    """
    quantum = _BLOCK_DEPTH * _MINIMUM_LOOP_BLOCKS
    asked_rows = max(_TILE, -(-matmul.rows // _TILE) * _TILE)
    asked_columns = max(_TILE, -(-matmul.columns // _TILE) * _TILE)
    machine_depth = max(quantum, -(-matmul.depth // quantum) * quantum)

    machine_rows = _placeable_rows(index=index, matmul=matmul, asked_rows=asked_rows, columns=asked_columns)
    if machine_rows != asked_rows and not matmul.epilogue <= _GELU_OPERATIONS:
        # Splitting the rows recomputes whatever stands after the matmul once per machine tile. An elementwise
        # transform reads one element and leaves that unchanged, but one reducing across the tile reads the whole
        # of it, so it would fold over a narrower tile than the kernel wrote and quietly answer a different sum.
        message = (
            f"Unable to run matmul {index}'s block of {matmul.rows} rows as machine tiles of {machine_rows}, "
            f"which is what the register file places, while an epilogue running "
            f"{', '.join(sorted(matmul.epilogue))} stands after it. Splitting the rows runs that epilogue once "
            f"per machine tile, so one reducing across the tile would fold over {machine_rows} rows rather than "
            f"the {matmul.rows} the kernel wrote. Narrow the kernel's own tile to what the file places."
        )
        raise ProgramError(message)

    return machine_depth // _BLOCK_DEPTH, machine_rows, asked_columns


def _placeable_rows(*, index: int, matmul: MatMul, asked_rows: int, columns: int) -> int:
    """Returns the rows one machine tile covers, being the widest the register file places.

    A tile the file cannot stage is run as several, so this is the largest placeable multiple of `_TILE` dividing
    what the kernel asked for, which leaves the launch a whole number of blocks to walk. Where no divisor places,
    the widest that does stands and the plan pads the residue.

    Args:
        index: Position of the matmul in the kernel.
        matmul: The matmul to judge.
        asked_rows: Rows the kernel's tile covers, rounded up to a whole tile.
        columns: Columns the kernel's tile covers, rounded up to a whole tile.

    Returns:
        The rows one machine tile covers.

    Raises:
        ProgramError: If the file does not place even a single tile of rows against these columns.
    """
    placeable: list[int] = []
    narrowest: LayoutError | None = None
    for rows in range(_TILE, asked_rows + 1, _TILE):
        try:
            register_map(block_m=rows, block_n=columns)
        except LayoutError as error:
            # Holds what the narrowest tile said, which is the reason a block of these columns places nowhere.
            narrowest = narrowest or error
            continue
        placeable.append(rows)

    if not placeable:
        message = (
            f"Unable to place matmul {index}'s block of {matmul.rows} by {matmul.columns}, which is {_TILE} by "
            f"{columns} at the narrowest the array computes. {narrowest}"
        )
        raise ProgramError(message) from narrowest

    divisors = [rows for rows in placeable if not asked_rows % rows]
    return max(divisors) if divisors else max(placeable)


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


def _stated_extent(*, index: int, axis: str, unit: int, sources: dict[str, int],
                   kernel_unit: int | None = None, grid_axes: frozenset[str] = frozenset()) -> int:
    """Returns what one axis covers, refusing a kernel whose loops, masks and caller do not agree on it.

    Notes:
        A launch issues whole tiles, so two sources agree when they round up to the same number of them and the
        tightest is what the padding is measured from. Sources rounding up differently name different launches.

    Args:
        index: Position of the matmul in the kernel.
        axis: The axis being read.
        unit: Positions along the axis one whole tile covers.
        sources: What each source says the axis covers, keyed by how it is named in a message.
        kernel_unit: Extent of the tile the kernel asked for, before the array's own tile stood in for it.
        grid_axes: Output axes the kernel's grid walks, which decide whether an axis nothing states is one the
            grid walks or one whose single trip loop was folded.

    Returns:
        The extent, being the tightest source.

    Raises:
        ProgramError: If nothing states the axis, or if two sources name different launches.
    """
    noun = _AXIS_NOUNS[axis]
    if not sources:
        # A loop running once leaves nothing behind, canonicalization folding it before the reader sees the
        # module, so a kernel stating this axis nowhere else covers exactly the tile it asked for. A grid walks
        # its axis outside the module and leaves the same silence behind while covering many of them, which no
        # tile extent recovers, so that one is refused rather than read as a single trip.
        if axis in grid_axes:
            message = (
                f"Unable to plan matmul {index} without the {noun} it covers. Nothing states it: its module "
                f"carries no loop walking {axis} and no mask bounding it, and its caller asked for no extent "
                f"either, while the kernel's grid walks {axis} itself. An axis put on the grid reaches a launch "
                f"rather than the module, so the tile it asks for is one program's share of the {noun} rather "
                f"than the whole of them: state that axis at the plan."
            )
            raise ProgramError(message)
        return kernel_unit if kernel_unit is not None else unit

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


def _resident_blocks(*, index: int, matmul: MatMul, covered: dict[str, int],
                     machine_rows: int = _BLOCK_TOKENS, machine_columns: int = _TILE) -> dict[str, int]:
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
        machine_rows: Rows one machine tile covers.
        machine_columns: Columns one machine tile covers.

    Returns:
        The blocks held live along "m" and along "n", each one where that axis walks outside the reduction.

    Raises:
        ProgramError: If the register file cannot stage the block those live sets imply.
    """
    # Reads the axes from the nest rather than from the steps, which two loops sharing a step leave ambiguous.
    live = {"m": 1, "n": 1}
    for axis in matmul.nest[matmul.nest.index(_REDUCTION_AXIS) + 1:]:
        if axis in live:
            live[axis] = covered[axis] // (machine_rows if axis == "m" else machine_columns)

    block_m, block_n = live["m"] * machine_rows, live["n"] * machine_columns
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
