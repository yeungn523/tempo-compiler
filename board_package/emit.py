"""Derives one GEMM accumulation step's bundle placement from the published ISA contract.
"""

from __future__ import annotations

import pathlib
import sys

# Places this package on sys.path, which `asm`, `bf16` and `isa_model` are imported from below and which is not
# installed, so this must precede those imports rather than sit beneath them.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collections.abc import Callable  # noqa: E402

import asm as asm_module  # noqa: E402
from bf16 import _NUM_SUBLANES, _TILE_ROWS, _TILE_VREGS, _TOKEN_GROUPS  # noqa: E402
from isa_model import MATRIX_OCCUPANCY, RTL_PARAMS  # noqa: E402

_WEIGHT_VREG: int = 24
"""First of the four vector registers a step stages its weight tile into."""

_ACTIVATION_VREG: int = 12
"""First of the vector registers a step stages its activation tiles into, one token group after another."""

_RESULT_VREG: int = 28
"""First of the four vector registers a matrix result pops into."""

_MATLOAD_AT: int = (_TILE_VREGS - 1) + RTL_PARAMS["WB_W_VLD"] + 1
"""Offset the weight load issues at, being the first cycle every weight register has landed.

The weight loads occupy the step's first `_TILE_VREGS` offsets, and a load's earliest dependent consumer sits one
cycle past its writeback reservation.
"""

_PUSH_AT: int = _MATLOAD_AT + MATRIX_OCCUPANCY["vmatload"] + 1
"""Offset the first activation push issues at, being the first cycle the matrix controller is free again.

A command issued at one cycle holds the controller for its occupancy, so the next becomes legal one cycle later.
"""

_PUSH_II: int = MATRIX_OCCUPANCY["vmatpush"] + 1
"""Offsets between consecutive pushes, being the interval the controller's occupancy allows."""

_POP_II: int = 5
"""Offsets between consecutive pops, which is scheduling policy rather than a consequence of the ISA.

A pop holds the controller for seven cycles, so the ISA alone would place the next one eight offsets later. Placing
them closer states an intent to issue and leaves the real distance to `asm.schedule`, which prices it from that same
occupancy. Widening this changes the image without changing what it computes.
"""

_ACCUMULATE_POP_GAP: int = 2
"""Offsets left after a group's accumulate before the next pop, which is scheduling policy.

The adds hold the result registers the following pop overwrites, so the gap keeps them clear of its write window.
"""


def emit_gemm_step(*, activation_row: int, weight_row: int, close_loop: bool) -> dict[int, dict[str, int]]:
    """Returns one accumulation step's bundles, keyed by their offset from the step's first bundle.

    Notes:
        One rule places the matrix commands. Push as early as the output FIFO allows and pop only when a further
        push would overflow it, because a pop stalls until the array produces the result it takes while a push is
        what the array waits on. The FIFO's depth is read rather than assumed, so a deeper one changes the
        placement without changing this function.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.

    Returns:
        The step's bundles, keyed by offset, each holding the slot words that bundle issues.
    """
    builder = asm_module.AsmBuilder()
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Merges slot words into one bundle, so a push and a load co-issue where both fit."""
        bundles.setdefault(offset, {}).update(slots)

    for vreg in range(_TILE_VREGS):
        put(offset=vreg,
            x=builder.vld(vreg=_WEIGHT_VREG + vreg, row=weight_row + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(offset=_TILE_VREGS + vreg,
            x=builder.vld(vreg=_ACTIVATION_VREG + vreg, row=activation_row + _NUM_SUBLANES * vreg))
    put(offset=_MATLOAD_AT, m=builder.vmatload(vreg=_WEIGHT_VREG))

    def stage_next(*, group: int, vreg: int) -> int | None:
        """Returns the load of the next token group's tile, or None once the last group is staged."""
        if group + 1 >= _TOKEN_GROUPS:
            return None
        return builder.vld(vreg=_ACTIVATION_VREG + _TILE_VREGS * (group + 1) + vreg,
                           row=activation_row + _TILE_ROWS * (group + 1) + _NUM_SUBLANES * vreg)

    _place_matrix_commands(put=put, builder=builder, stage_next=stage_next, close_loop=close_loop)
    return bundles


def step_stream(*, activation_row: int, weight_row: int, close_loop: bool) -> tuple[int, ...]:
    """Returns one accumulation step as the contiguous bundle words it places, zero filling the offsets it leaves.

    Args:
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether the step's final accumulate carries the enclosing loop's closing bundle.

    Returns:
        The bundle words, in placement order.
    """
    bundles = emit_gemm_step(activation_row=activation_row, weight_row=weight_row, close_loop=close_loop)
    words = []
    for offset in range(max(bundles) + 1):
        word = 0
        for slot_word in bundles.get(offset, {}).values():
            word |= slot_word
        words.append(word)
    return tuple(words)


def _place_matrix_commands(*, put: Callable[..., None], builder: asm_module.AsmBuilder,
                           stage_next: Callable[..., int | None], close_loop: bool) -> None:
    """Places every push, pop and accumulate of one step, greedily against the matrix output FIFO's depth.

    Args:
        put: Merges slot words into the bundle at one offset.
        builder: Encodes each slot word.
        stage_next: Returns the load riding under one push, or None where there is nothing left to stage.
        close_loop: Determines whether the final accumulate carries the enclosing loop's closing bundle.
    """
    depth, per_push, per_pop = asm_module.mxu_output_fifo()
    at = _PUSH_AT
    outstanding = 0
    popped = 0
    pending = -1

    def flush_accumulate() -> None:
        """Places a drained tile's adds, held back so they never split a run of pushes."""
        nonlocal at, pending
        if pending < 0:
            return
        last = pending == _TOKEN_GROUPS - 1
        for vreg in range(_TILE_VREGS):
            accumulator = _TILE_VREGS * pending + vreg
            put(offset=at, e=builder.vadd(vd=accumulator, va=accumulator, vb=_RESULT_VREG + vreg))
            if close_loop and last and vreg == _TILE_VREGS - 1:
                put(offset=at, f=builder.loop_end())
            at += 1
        if not last:
            at += _ACCUMULATE_POP_GAP
        pending = -1

    def drain_one() -> None:
        """Pops the oldest un-drained result, marking a tile ready once all of its registers have landed."""
        nonlocal at, outstanding, popped, pending
        vreg = popped % _TILE_VREGS
        if vreg == 0:
            # The adds read the result registers this pop is about to overwrite.
            flush_accumulate()
        put(offset=at, m=builder.vmatpop(vreg=_RESULT_VREG + vreg))
        at += _POP_II
        outstanding -= per_pop
        popped += 1
        if vreg == _TILE_VREGS - 1:
            pending = popped // _TILE_VREGS - 1

    for group in range(_TOKEN_GROUPS):
        for vreg in range(_TILE_VREGS):
            while outstanding + per_push > depth:
                drain_one()
            slots = {"m": builder.vmatpush(vreg=_ACTIVATION_VREG + _TILE_VREGS * group + vreg)}
            staged = stage_next(group=group, vreg=vreg)
            if staged is not None:
                slots["x"] = staged
            put(offset=at, **slots)
            at += _PUSH_II
            outstanding += per_push
    while outstanding > 0:
        drain_one()
    flush_accumulate()
