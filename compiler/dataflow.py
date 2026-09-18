"""Provides the dataflow one trip of an attention block's key loop runs."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import MatMul, Operation, ProgramError

CARRIED: str = "carried"
"""Role of a value the key loop hands from each trip to the next, which holds a dedicated register region."""

STREAMED: str = "streamed"
"""Role of a tile a trip reads from memory, which a load places rather than an allocator."""

CONSTANT: str = "constant"
"""Role of a number the kernel states, which reaches the vector unit as a tile a host stages."""

TRANSIENT: str = "transient"
"""Role of a value one trip computes and consumes, which shares the register region the phases pass through."""

_ACCUMULATOR: str = "accumulator"
"""Region the output tiles the second product accumulates into hold."""

_ROW_MAX: str = "row_max"
"""Region the running row maxima hold, being the statistic the probabilities are taken against."""

_ROW_SUM: str = "row_sum"
"""Region the running row sums hold, being the statistic the output is normalized by."""


@dataclass(frozen=True, slots=True)
class Value:
    """One value a trip names, and where the register file holds it.

    Attributes:
        identifier: Identifier the module gives the value.
        role: Whether the value crosses trips, arrives from memory, states a number, or lives inside one trip.
        region: Register region a carried value holds, None for every other role.
        number: Number a constant carries, None for every other role.
    """

    identifier: int
    role: str
    region: str | None
    number: float | None = None


@dataclass(frozen=True, slots=True)
class Trip:
    """One trip of an attention block's key loop, as the operations it runs over the values they name.

    The chains a matmul carries overlap, because Triton folds one value's update onto another's, so a lowering
    reads the merged order here rather than walking each chain apart.

    Attributes:
        operations: Every operation one trip runs, in program order, the two products among them.
        after: The operations the loop's result passes through once the loop ends.
        values: Role of every value the trip names, keyed by identifier.
        carried: Identifier a region holds entering one trip and the one it holds leaving it, keyed by region.
    """

    operations: tuple[Operation, ...]
    after: tuple[Operation, ...]
    values: dict[int, Value]
    carried: dict[str, tuple[int, int]]

    @property
    def constants(self) -> tuple[float, ...]:
        """Returns every number wanting a staged tile of its own, each appearing once however often it is read.

        Zero is not among them, an attention block already staging a tile of it to clear its accumulators from.
        """
        stated = [value.number for value in self.values.values()
                  if value.role == CONSTANT and value.number is not None and value.number != 0.0]
        return tuple(dict.fromkeys(stated))


def attention_trip(matmuls: tuple[MatMul, ...], *, produces: int, consumes: int) -> Trip:
    """Returns the dataflow one trip of an attention block's key loop runs.

    Args:
        matmuls: The matmuls a kernel computes, in program order.
        produces: Position of the matmul leaving the scores.
        consumes: Position of the matmul contracting the probabilities.

    Returns:
        The trip.

    Raises:
        ProgramError: If either matmul carries no record of its own product, or if a value the loop carries
            holds no role the register file has a region for.
    """
    scores, attended = matmuls[produces], matmuls[consumes]
    operations = _trip_operations(scores=scores, attended=attended)
    running = {operation.position for operation in operations}
    after = tuple(one for one in attended.epilogue_chain if one.position not in running)
    regions = _carried_regions(attended=attended)
    return Trip(operations=operations, after=after,
                values=_values(operations, attended=attended, regions=regions,
                               numbers=dict(attended.constants)),
                carried={regions[one.position]: (one.inside, one.leaving) for one in attended.carried})


def _trip_operations(*, scores: MatMul, attended: MatMul) -> tuple[Operation, ...]:
    """Returns every operation one trip runs, merged from the chains that overlap over it.

    Args:
        scores: The matmul leaving the scores.
        attended: The matmul contracting the probabilities.

    Returns:
        The operations in program order, with each appearing once however many chains reach it.

    Raises:
        ProgramError: If either matmul carries no record of its own product.
    """
    found: dict[int, Operation] = {}
    for stated, matmul in (("leaving the scores", scores), ("contracting the probabilities", attended)):
        if matmul.product is None:
            message = (
                f"Unable to read the trip of an attention block whose matmul {stated} carries no record of its "
                f"own product. The chains around a matmul terminate at it rather than naming it, so the record "
                f"is what joins them into one graph."
            )
            raise ProgramError(message)
        found[matmul.product.position] = matmul.product
    chains = (attended.prologue_chain, attended.accumulator_chain,
              *(carried.chain for carried in attended.carried))
    for chain in chains:
        for operation in chain:
            found[operation.position] = operation
    return tuple(sorted(found.values(), key=lambda one: one.position))


def _carried_regions(*, attended: MatMul) -> dict[int, str]:
    """Returns the register region each carried value holds, read from what the trip does with it.

    The accumulator is the value the second product accumulates onto, and the running maximum is the statistic
    the probabilities are taken against, so both are named by use rather than by the arithmetic updating them.

    Args:
        attended: The matmul contracting the probabilities.

    Returns:
        The region each carried value holds, keyed by its position in the loop.

    Raises:
        ProgramError: If the trip leaves a carried value unnamed, which is one the register file holds nowhere.
    """
    named: dict[int, str] = {}
    reads = {identifier for one in attended.prologue_chain for identifier in one.operands}
    accumulated = None if attended.product is None else attended.product.result
    for carried in attended.carried:
        if carried.leaving == accumulated:
            named[carried.position] = _ACCUMULATOR
        elif {carried.inside, carried.leaving} & reads:
            named[carried.position] = _ROW_MAX
    unnamed = [carried for carried in attended.carried if carried.position not in named]
    if len(unnamed) == 1 and set(named.values()) == {_ACCUMULATOR, _ROW_MAX}:
        named[unnamed[0].position] = _ROW_SUM
        return named

    stated = ", ".join(f"{position} as {region}" for position, region in sorted(named.items())) or "none of them"
    message = (
        f"Unable to name every value the key loop carries, of which this trip names {stated}. An attention "
        f"block holds an accumulator the second product accumulates onto, a running maximum the probabilities "
        f"are taken against and a running sum, and the register file holds a value outside those nowhere."
    )
    raise ProgramError(message)


def _values(operations: tuple[Operation, ...], *, attended: MatMul, regions: dict[int, str],
            numbers: dict[int, float]) -> dict[int, Value]:
    """Returns the role every value a trip names holds.

    Args:
        operations: Every operation the trip runs, in program order.
        attended: The matmul contracting the probabilities.
        regions: Region each carried value holds, keyed by its position in the loop.
        numbers: Number each constant the kernel states carries, keyed by the value it defines.

    Returns:
        The role of each value, keyed by identifier.
    """
    found: dict[int, Value] = {}
    for carried in attended.carried:
        for identifier in (carried.entering, carried.inside, carried.leaving, carried.result):
            found[identifier] = Value(identifier=identifier, role=CARRIED, region=regions[carried.position],
                                      number=numbers.get(carried.entering))
    produced = {operation.result for operation in operations}
    for operation in operations:
        found.setdefault(operation.result, Value(identifier=operation.result, role=TRANSIENT, region=None))
        for operand in operation.operands:
            if operand in produced:
                found.setdefault(operand, Value(identifier=operand, role=TRANSIENT, region=None))
                continue
            number = numbers.get(operand)
            role = STREAMED if number is None else CONSTANT
            found.setdefault(operand, Value(identifier=operand, role=role, region=None, number=number))
    return found
