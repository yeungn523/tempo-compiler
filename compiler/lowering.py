"""Provides the lowering of a read kernel's operations into the steps an image runs."""

from __future__ import annotations

import collections
import dataclasses

from bf16 import _VREG_ELEMENTS
from emit import (POINTWISE_LEFT, POINTWISE_RIGHT, Accumulate, Combine, Load, Product, RowLoad, RowReduce, RowStore,
                  Step, Transform, VectorOp)
from isa_model import PROFILE

from .allocation import COMMIT, Place, Placement
from .dataflow import CONSTANT, Trip
from .ir import Operation, Pointwise, ProgramError, RowTerm, Rows

_COMBINERS: dict[str, str] = {
    "arith.addf": "vadd",
    "arith.subf": "vsub",
    "arith.mulf": "vmul",
    "arith.maxnumf": "vmax",
    "arith.minnumf": "vmin",
}
"""Vector operation each elementwise arithmetic operation runs as, keyed by the name a module prints."""

_TRANSFORMS: dict[str, str] = {
    "math.exp": "vexp",
    "math.rsqrt": "vrsqrt",
}
"""Vector operation each single operand arithmetic operation runs as, keyed by the name a module prints."""

_REDUCTIONS: dict[str, str] = {
    "tt.reduce.arith.maxnumf": "vmax",
    "tt.reduce.arith.addf": "vadd",
}
"""Vector operation each reduce folds its rows under, keyed by the combiner the reduce carries."""

_RECIPROCAL: str = "arith.divf"
"""Division, which the machine runs as a reciprocal and a multiply because it carries no divider."""

_INVERSE: str = "reciprocal"
"""Name a row term's reciprocal is rewritten under, which no module prints and the vector unit runs as vrecip."""

_PRODUCT: str = "tt.dot"
"""Name the module prints a matrix product under, which the array runs rather than the vector unit."""

_ACCUMULATOR: str = "accumulator"
"""Region the output tiles hold, whose tiles are updated one at a time because some of them lie in VMEM."""

_RECIPROCAL_REGION: str = "staged"
"""Slot the normalizing reciprocal is taken into, every slot standing free once the key loop has ended."""

_ZERO_REGION: str = "zero"
"""Region the tile of zeros lies in, which an attention block stages to clear its accumulators from."""

_CONSTANT_REGION: str = "constant"
"""Region the tile of every other stated number lies in, one tile each in the order the trip states them."""


def lower_operation(operation: Operation, *, into: str, sources: tuple[str, ...]) -> tuple[Step, ...]:
    """Returns the steps one operation runs, refusing one the machine has no vector operation for.

    Args:
        operation: The operation to lower.
        into: Register region the result lands in.
        sources: Register region of each value the operation reads, in the order it reads them.

    Returns:
        The steps, being more than one where the machine spells the operation as several.

    Raises:
        ProgramError: If the machine runs no vector operation for this one, or if the regions given do not match
            the operands the operation reads.
    """
    name = operation.name
    if name in _COMBINERS:
        _require(operation, sources=sources, wanted=2)
        return (Combine(kind=_COMBINERS[name], into=into, left=sources[0], right=sources[1]),)
    if name in _TRANSFORMS:
        _require(operation, sources=sources, wanted=1)
        return (Transform(kind=_TRANSFORMS[name], into=into, source=sources[0]),)
    if name in _REDUCTIONS:
        _require(operation, sources=sources, wanted=1)
        return (RowReduce(kind=_REDUCTIONS[name], source=sources[0], into=into),)
    if name == _RECIPROCAL:
        _require(operation, sources=sources, wanted=2)
        return (Transform(kind="vrecip", into=into, source=sources[1]),
                Combine(kind="vmul", into=into, left=sources[0], right=into))

    message = (
        f"Unable to name {name} as a step. The vector unit runs "
        f"{', '.join(sorted(_COMBINERS | _TRANSFORMS | _REDUCTIONS))} and a division through its reciprocal, so "
        f"an operation outside those is one this machine computes in no single pass. Write it in terms of those, "
        f"or compute it in a further launch."
    )
    raise ProgramError(message)


def attention_body(placement: Placement, *, trip: Trip, steps: int, tiles: int) -> tuple[Step, ...]:
    """Returns the steps one trip of an attention block's key loop runs.

    Args:
        placement: The order and the regions the trip's operations were placed under.
        trip: The dataflow the placement was made of.
        steps: Tiles the score product contracts over, being the head over the tile.
        tiles: Tiles the head covers, which is what the accumulator spans.

    Returns:
        The steps, in the order they run.

    Raises:
        ProgramError: If a placed operation is one no step runs.
    """
    attended = _attended(placement)
    body: list[Step] = []
    for place in placement.places:
        if place.operation.name == COMMIT:
            body.append(Transform(kind="vmov", into=place.into, source=place.sources[0]))
        elif place.operation.name != _PRODUCT:
            body.extend(_stage_constants(place, trip=trip))
            body.extend(_lower_placed(place, tiles=tiles))
        elif place is attended:
            body.extend(_attended_product(place, tiles=tiles))
        else:
            body.extend(_score_product(place, steps=steps))
    return tuple(body)


def attention_epilogue(trip: Trip, *, tiles: int) -> tuple[Step, ...]:
    """Returns the steps an attention block runs once its key loop has ended.

    Args:
        trip: The dataflow one trip runs, whose operations after the loop are normalized here.
        tiles: Tiles the head covers, which is what the accumulator spans.

    Returns:
        The steps, in the order they run.

    Raises:
        ProgramError: If the block ends on anything other than dividing its accumulator by its running sum.
    """
    divides = len(trip.after) == 1 and trip.after[0].name == _RECIPROCAL
    regions = [trip.values[operand].region for operand in trip.after[0].operands] if divides else []
    if regions == [_ACCUMULATOR, "row_sum"]:
        return (Transform(kind="vrecip", into=_RECIPROCAL_REGION, source="row_sum"),
                *(Accumulate(kind="vmul", tile=tile, operand=_RECIPROCAL_REGION, store=True)
                  for tile in range(tiles)))

    stated = ", ".join(one.name for one in trip.after) or "nothing"
    message = (
        f"Unable to drain an attention block ending on {stated}. The running sum is what the accumulated "
        f"output is normalized by, and the machine divides through a reciprocal, so a block ending on anything "
        f"else is one this drain does not spell."
    )
    raise ProgramError(message)


def unnameable(chain: tuple[Operation, ...]) -> tuple[str, ...]:
    """Returns the name of every operation in a chain the machine has no vector operation for.

    Args:
        chain: The operations to judge, in the order they run.

    Returns:
        The names, empty where every operation lowers.
    """
    known = _COMBINERS.keys() | _TRANSFORMS.keys() | _REDUCTIONS.keys() | {_RECIPROCAL}
    return tuple(dict.fromkeys(step.name for step in chain if step.name not in known))


def attention_block(matmuls: tuple, *, index: int) -> bool:
    """Returns whether one matmul reads a product through the online softmax an attention block carries.

    Args:
        matmuls: The matmuls a kernel computes, in program order.
        index: Position of the matmul to judge.

    Returns:
        True when the matmul contracts probabilities its own prologue exponentiated.
    """
    matmul = matmuls[index]
    if matmul.left_source is None or not matmul.carried:
        return False
    chain = {step.name for step in matmul.prologue_chain}
    return bool(chain & _REDUCTIONS.keys()) and bool(chain & _TRANSFORMS.keys())


def attention_extents(matmuls: tuple, *, produces: int, consumes: int) -> dict[str, int]:
    """Returns the extents one attention image covers, read from the pair it runs.

    Args:
        matmuls: The matmuls a kernel computes, in program order.
        produces: Position of the matmul leaving the scores.
        consumes: Position of the matmul contracting the probabilities.

    Returns:
        The query blocks a host issues, the blocks the key loop walks, the queries one block covers and the
        width of the head.

    Raises:
        ProgramError: If the kernel states neither the queries nor the keys its loops walk.
    """
    scores, attended = matmuls[produces], matmuls[consumes]
    queries, head = scores.rows, attended.columns
    walked = {loop.axis: loop for loop in scores.tile_loops}
    if "m" not in walked or "n" not in walked:
        stated = ", ".join(sorted(axis for axis in walked if axis)) or "neither axis"
        message = (
            f"Unable to read the extents matmul {produces} covers, whose loops state {stated}. An attention "
            f"image walks the keys itself and a host issues it per query block, so both have to be loops this "
            f"reader can name."
        )
        raise ProgramError(message)
    return {"row_blocks": -(-walked["m"].bound // queries), "key_blocks": walked["n"].trips,
            "queries": queries, "head": head}


def pointwise_body(pointwise: Pointwise) -> tuple[VectorOp, ...]:
    """Returns the vector operations one VREG of a pointwise image runs, every value holding a register of its own.

    Args:
        pointwise: The kernel the reader recovered.

    Returns:
        The operations in program order, the last one leaving the result the image stores.

    Raises:
        ProgramError: If an operation has no vector operation to run as, or the values outnumber the register file.
    """
    left, right = pointwise.operands
    registers = {left: POINTWISE_LEFT, right: POINTWISE_RIGHT}
    body: list[VectorOp] = []
    for operation in pointwise.chain:
        into = len(registers)
        body.extend(_vector_ops(operation.name, into=into,
                                sources=tuple(registers[operand] for operand in operation.operands), image="pointwise"))
        registers[operation.result] = into
    if len(registers) > PROFILE["vreg_entry_count"]:
        message = (
            f"Unable to hold a pointwise body's {len(registers)} values in {PROFILE['vreg_entry_count']} VREGs. "
            f"Every value keeps a register of its own, so shorten the arithmetic one store computes."
        )
        raise ProgramError(message)
    return tuple(body)


def row_body(rows: Rows) -> tuple[tuple[float, ...], tuple[RowLoad, ...], tuple[RowLoad | VectorOp | RowStore, ...]]:
    """Returns the operations a row image runs, every register assigned.

    Notes:
        Each sum runs as a pass over the token's VREGs, reducing every one with `vredsum` and adding the partials,
        which leaves the total broadcast across a VREG. The output pass then stores one VREG at a time. A value one
        token shares is computed once, before the first pass reading it, while a per-element value is recomputed in
        every pass that reads it, since the row's VREGs outnumber the registers that could hold it.

    Args:
        rows: The kernel the reader recovered.

    Returns:
        The number each broadcast constant holds, the loads taking them into registers before the token loop, and
        the operations one token runs.

    Raises:
        ProgramError: If an operation has no vector operation to run as, or the live values outnumber the registers.
    """
    tiles = rows.features // _VREG_ELEMENTS
    fresh = iter(range(1 << 30))
    scalars: list[float] = []
    prologue: list[RowLoad] = []
    body: list[RowLoad | VectorOp | RowStore] = []
    token_values: dict[RowTerm, int] = {}

    def token_value(term: RowTerm) -> int:
        """Returns the register holding one per-token term, computing it on first use."""
        if term in token_values:
            return token_values[term]
        if term.name == "constant":
            into = next(fresh)
            prologue.append(RowLoad(into=into, region="scalar", index=len(scalars), tile=0))
            scalars.append(float(term.value))
        elif term.name == "sum":
            into = total(term.operands[0])
        else:
            into = next(fresh)
            body.extend(_vector_ops(term.name, into=into, sources=tuple(token_value(operand)
                                                                        for operand in term.operands), image="row"))
        token_values[term] = into
        return into

    def prepare(term: RowTerm) -> None:
        """Computes every per-token term a per-element one reads, ahead of the pass reading them."""
        if term.per_token:
            token_value(term)
            return
        for operand in term.operands:
            prepare(operand)

    def element(term: RowTerm, *, tile: int, computed: dict[RowTerm, int]) -> int:
        """Returns the register holding one term at one VREG of the row, computing what this pass has not."""
        if term.per_token:
            return token_values[term]
        if term in computed:
            return computed[term]
        into = next(fresh)
        if term.name == "input":
            body.append(RowLoad(into=into, region="input", index=0, tile=tile))
        elif term.name == "row":
            body.append(RowLoad(into=into, region="row", index=rows.rows.index(term.value), tile=tile))
        else:
            sources = tuple(element(operand, tile=tile, computed=computed) for operand in term.operands)
            body.extend(_vector_ops(term.name, into=into, sources=sources, image="row"))
        computed[term] = into
        return into

    def total(term: RowTerm) -> int:
        """Returns the register holding a term's total over the row, broadcast, emitting the pass computing it."""
        prepare(term)
        partials = []
        for tile in range(tiles):
            partial = next(fresh)
            body.append(VectorOp(kind="vredsum", into=partial, sources=(element(term, tile=tile, computed={}),)))
            partials.append(partial)
        running = partials[0]
        for partial in partials[1:]:
            into = next(fresh)
            body.append(VectorOp(kind="vadd", into=into, sources=(running, partial)))
            running = into
        return running

    output = _fold_reciprocals(rows.output, folded={})
    prepare(output)
    for tile in range(tiles):
        body.append(RowStore(source=element(output, tile=tile, computed={}), tile=tile))
    prologue_registers, body_registers = _assign_registers(prologue=tuple(prologue), body=tuple(body))
    return tuple(scalars), prologue_registers, body_registers


def _vector_ops(name: str, *, into: int, sources: tuple[int, ...], image: str) -> tuple[VectorOp, ...]:
    """Returns the vector operations one elementwise operation runs as, the result landing in `into`.

    Args:
        name: The operation's name, as the module prints it.
        into: The VREG the result lands in.
        sources: The VREGs the operation reads, in operand order.
        image: The kind of image being lowered, which the diagnostic names.

    Returns:
        The operations, being two for a division, which runs as the divisor's reciprocal times the dividend.

    Raises:
        ProgramError: If the vector unit runs nothing for the operation over that many operands.
    """
    if name in _COMBINERS and len(sources) == 2:
        return (VectorOp(kind=_COMBINERS[name], into=into, sources=sources),)
    if name in _TRANSFORMS and len(sources) == 1:
        return (VectorOp(kind=_TRANSFORMS[name], into=into, sources=sources),)
    if name == _INVERSE and len(sources) == 1:
        return (VectorOp(kind="vrecip", into=into, sources=sources),)
    if name == _RECIPROCAL and len(sources) == 2:
        # Holds the reciprocal in the result's register, which the multiply then overwrites in place.
        return (VectorOp(kind="vrecip", into=into, sources=(sources[1],)),
                VectorOp(kind="vmul", into=into, sources=(sources[0], into)))
    message = (
        f"Unable to lower {name} over {len(sources)} operands in a {image} image. It runs "
        f"{', '.join(sorted({*_COMBINERS, *_TRANSFORMS, _RECIPROCAL}))} and nothing else."
    )
    raise ProgramError(message)


def _fold_reciprocals(term: RowTerm, *, folded: dict[RowTerm, RowTerm]) -> RowTerm:
    """Returns a term with every division by a value one token shares rewritten as a product with its reciprocal.

    The host computes a stated number's reciprocal at full precision when it packs the constant, which the vector
    unit's own reciprocal would only approximate. Any other per-token divisor is reciprocated once per token rather
    than once per VREG of the row.

    Args:
        term: The term to rewrite.
        folded: The rewrite of every term already visited.

    Returns:
        The rewritten term.
    """
    if term in folded:
        return folded[term]
    operands = tuple(_fold_reciprocals(operand, folded=folded) for operand in term.operands)
    if term.name == _RECIPROCAL and operands[1].name == "constant":
        rewritten = RowTerm(name="arith.mulf",
                            operands=(operands[0], RowTerm(name="constant", value=1.0 / float(operands[1].value))))
    elif term.name == _RECIPROCAL and operands[1].per_token:
        rewritten = RowTerm(name="arith.mulf", operands=(operands[0], RowTerm(name=_INVERSE, operands=operands[1:])))
    else:
        rewritten = RowTerm(name=term.name, operands=operands, value=term.value)
    folded[term] = rewritten
    return rewritten


def _assign_registers(*, prologue: tuple[RowLoad, ...], body: tuple[RowLoad | VectorOp | RowStore, ...]
                      ) -> tuple[tuple[RowLoad, ...], tuple[RowLoad | VectorOp | RowStore, ...]]:
    """Returns a row image's operations with every value given a VREG, each one freed after its last read.

    Notes:
        The prologue's constants keep their registers for the whole image. A freed register rejoins the back of the
        queue, which spaces its reuse out and leaves the packing pass fewer false dependencies to order.

    Args:
        prologue: The constant loads, naming values rather than registers.
        body: The operations one token runs, naming values rather than registers.

    Returns:
        The same operations, naming registers.

    Raises:
        ProgramError: If more values are live at once than the register file holds.
    """
    registers = {load.into: index for index, load in enumerate(prologue)}
    pinned = set(registers)
    free = collections.deque(range(len(registers), PROFILE["vreg_entry_count"]))
    last: dict[int, int] = {}
    for position, operation in enumerate(body):
        for value in _reads(operation):
            last[value] = position

    assigned: list[RowLoad | VectorOp | RowStore] = []
    for position, operation in enumerate(body):
        reads = _reads(operation)
        for value in dict.fromkeys(reads):
            if value not in pinned and last[value] == position:
                free.append(registers[value])
        if isinstance(operation, RowStore):
            assigned.append(dataclasses.replace(operation, source=registers[operation.source]))
            continue
        if operation.into not in registers:
            if not free:
                message = (
                    f"Unable to hold a row image's live values in {PROFILE['vreg_entry_count']} VREGs. Each sum keeps "
                    f"one partial per VREG of the row, so sum a narrower row or fewer values at once."
                )
                raise ProgramError(message)
            registers[operation.into] = free.popleft()
        into = registers[operation.into]
        if operation.into not in last:
            free.append(into)
        if isinstance(operation, RowLoad):
            assigned.append(dataclasses.replace(operation, into=into))
        else:
            assigned.append(dataclasses.replace(operation, into=into,
                                                sources=tuple(registers[value] for value in reads)))
    return tuple(dataclasses.replace(load, into=registers[load.into]) for load in prologue), tuple(assigned)


def _reads(operation: RowLoad | VectorOp | RowStore) -> tuple[int, ...]:
    """Returns the values one row image operation reads."""
    if isinstance(operation, RowStore):
        return (operation.source,)
    if isinstance(operation, VectorOp):
        return operation.sources
    return ()


def _require(operation: Operation, *, sources: tuple[str, ...], wanted: int) -> None:
    """Refuses a lowering given other regions than the operation reads operands.

    Args:
        operation: The operation being lowered.
        sources: Register region of each value it reads.
        wanted: Operands the machine's spelling of this operation takes.

    Raises:
        ProgramError: If the count given is not the count wanted.
    """
    if len(sources) == wanted:
        return
    message = (
        f"Unable to lower {operation.name} from {len(sources)} regions, which takes {wanted}. The regions state "
        f"where each operand the operation reads lives, so one per operand is what names them."
    )
    raise ProgramError(message)


def _attended(placement: Placement) -> Place:
    """Returns the product accumulating onto the output, being the one leaving a value in its region.

    Args:
        placement: The order and the regions the trip's operations were placed under.

    Returns:
        The place the product holds.

    Raises:
        ProgramError: If no product accumulates onto the output, which is a trip holding no attention block.
    """
    found = [place for place in placement.places
             if place.operation.name == _PRODUCT and place.into == _ACCUMULATOR]
    if len(found) == 1:
        return found[0]
    message = (
        f"Unable to name the product attending over the values, of which {len(found)} accumulate onto the "
        f"output. An attention block runs one product leaving the scores and one accumulating what the online "
        f"softmax leaves onto the output, so exactly one of the pair writes the accumulator."
    )
    raise ProgramError(message)


def _lower_placed(place: Place, *, tiles: int) -> tuple[Step, ...]:
    """Returns the steps one placed operation runs, spreading a write of the output over its tiles.

    Args:
        place: The operation, the region its result lands in and the region of each value it reads.
        tiles: Tiles the head covers, which is what the accumulator spans.

    Returns:
        The steps.

    Raises:
        ProgramError: If the operation writes the output under something other than a combining operation.
    """
    if place.into != _ACCUMULATOR:
        return lower_operation(place.operation, into=place.into, sources=place.sources)
    kind = _COMBINERS.get(place.operation.name)
    operand = next((source for source in place.sources if source != _ACCUMULATOR), None)
    if kind is not None and operand is not None:
        return tuple(Accumulate(kind=kind, tile=tile, operand=operand) for tile in range(tiles))

    message = (
        f"Unable to write the output under {place.operation.name}. Some of the output lies in VMEM rather than "
        f"in registers, so it is updated one tile at a time under an operation combining it with one other "
        f"value, and this operation is not one of those."
    )
    raise ProgramError(message)


def _score_product(place: Place, *, steps: int) -> tuple[Step, ...]:
    """Returns the steps the product leaving the scores runs, one group per tile of the contraction.

    Args:
        place: The product, the region its result lands in and the slot each operand is staged in.
        steps: Tiles the product contracts over, being the head over the tile.

    Returns:
        The steps.
    """
    pushed, weight, cleared = place.sources
    body: list[Step] = [Load(into=cleared, region="zero")]
    for step in range(steps):
        body += [Load(into=weight, region="key", tile=step),
                 Load(into=pushed, region="query", tile=step),
                 Product(pushed=pushed, weight=weight),
                 Combine(kind="vadd", into=place.into, left=place.into, right="result")]
    return tuple(body)


def _attended_product(place: Place, *, tiles: int) -> tuple[Step, ...]:
    """Returns the steps the product attending over the values runs, one group per tile of the head.

    Args:
        place: The product, the region its result lands in and the slot each operand is staged in.
        tiles: Tiles the head covers, which is what the accumulator spans.

    Returns:
        The steps.
    """
    pushed, weight, _ = place.sources
    body: list[Step] = []
    for tile in range(tiles):
        body += [Load(into=weight, region="value", tile=tile),
                 Product(pushed=pushed, weight=weight),
                 Accumulate(kind="vadd", tile=tile, operand="result")]
    return tuple(body)


def _stage_constants(place: Place, *, trip: Trip) -> tuple[Step, ...]:
    """Returns the loads placing every number one operation reads into the slot it reads it from.

    Args:
        place: The operation, the region its result lands in and the region of each value it reads.
        trip: The dataflow the placement was made of, which states what number each value carries.

    Returns:
        The loads, empty where the operation reads no stated number.
    """
    staged: list[Step] = []
    for operand, source in zip(place.operation.operands, place.sources, strict=True):
        value = trip.values.get(operand)
        if value is None or value.role != CONSTANT or value.number is None:
            continue
        region, tile = ((_ZERO_REGION, 0) if value.number == 0.0
                        else (_CONSTANT_REGION, trip.constants.index(value.number)))
        staged.append(Load(into=source, region=region, tile=tile))
    return tuple(staged)
