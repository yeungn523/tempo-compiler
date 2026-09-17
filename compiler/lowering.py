"""Provides the lowering of a read kernel's operations into the steps an image runs."""

from __future__ import annotations

from emit import Combine, RowReduce, Step, Transform

from .ir import Operation, ProgramError

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
    "math.sqrt": "vrsqrt",
}
"""Vector operation each single operand arithmetic operation runs as, keyed by the name a module prints."""

_REDUCTIONS: dict[str, str] = {
    "tt.reduce.arith.maxnumf": "vmax",
    "tt.reduce.arith.addf": "vadd",
}
"""Vector operation each reduce folds its rows under, keyed by the combiner the reduce carries."""

_RECIPROCAL: str = "arith.divf"
"""Division, which the machine runs as a reciprocal and a multiply because it carries no divider."""


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
