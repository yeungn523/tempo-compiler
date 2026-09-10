"""Provides the register renaming check the emitter tests hold a bundle stream to.

The emitter lays its registers out from `layout.register_map` while `kernels` names its own, so a stream that is
right differs from the hand-written one in those names and nothing else.
"""

from __future__ import annotations

import asm


def renaming_onto(derived: tuple[int, ...], hand: tuple[int, ...]) -> dict[int, int] | None:
    """Returns the register renaming carrying one bundle stream onto another, or None when none does.

    The emitter lays its registers out from the map while the hand-written builder names its own, so the two agree
    on every slot a bundle occupies and on which registers it names together, but not on the numbers themselves.
    A single consistent renaming across the whole stream is what says the difference is only those names.

    Args:
        derived: The emitted stream.
        hand: The stream it is held to.

    Returns:
        The renaming, or None when the streams differ in anything a renaming cannot account for.
    """
    if len(derived) != len(hand):
        return None
    renaming: dict[int, int] = {}
    taken: dict[int, int] = {}
    for left, right in zip(derived, hand):
        first, second = asm._decode(left), asm._decode(right)
        for field in ("m_op", "f_op", "dma", "port_a", "vmem_cycles", "dma_slot", "flush_mask"):
            if first[field] != second[field]:
                return None
        if len(first["reads"]) != len(second["reads"]) or len(first["writes"]) != len(second["writes"]):
            return None
        pairs = list(zip(sorted(first["reads"]), sorted(second["reads"])))
        pairs += [(a, b) for (a, _), (b, _) in zip(first["writes"], second["writes"])]
        for source, target in pairs:
            if renaming.setdefault(source, target) != target or taken.setdefault(target, source) != source:
                return None
    return renaming
