"""Contains tests for the register file map a block's tiling derives, and the renaming check it implies.

The map exists so a tile size larger than the register file costs VMEM traffic rather than a refusal, so the tests
cover what stays resident as the block grows and what the file cannot stage at all.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

import asm  # noqa: E402
from isa_model import PROFILE  # noqa: E402
from layout import LayoutError, register_map  # noqa: E402


class RegisterMapTest(unittest.TestCase):
    """Covers the register map across the block shapes a kernel may ask for."""

    def test_the_regions_are_disjoint_and_fit_the_file(self) -> None:
        """Asserts no two regions overlap and the map never hands out a register the file lacks."""
        entries = PROFILE["vreg_entry_count"]
        for block_m in (16, 32, 48, 64, 80):
            for block_n in (16, 32, 64):
                with self.subTest(block_m=block_m, block_n=block_n):
                    mapped = register_map(block_m=block_m, block_n=block_n)
                    per_tile = mapped.token_groups * 4
                    spans = [(mapped.accumulator, mapped.resident * per_tile),
                             (mapped.activation, per_tile), (mapped.weight, 4), (mapped.result, 4)]
                    if mapped.scratch is not None:
                        spans.append((mapped.scratch, mapped.scratch_tiles * 4))
                    occupied: set[int] = set()
                    for base, width in spans:
                        registers = set(range(base, base + width))
                        self.assertFalse(registers & occupied, "regions overlap")
                        occupied |= registers
                    self.assertLessEqual(mapped.used, entries)
                    self.assertLessEqual(max(occupied), entries - 1)

    def test_a_block_wider_than_the_file_spills_rather_than_refusing(self) -> None:
        """Asserts output tiles past what the registers hold accumulate in VMEM instead of raising.

        Every tile spills once any does, because a spilled tile drains as one partial per VMEM bank and a resident
        one as a single tile, and a drain carrying both shapes is one the host cannot read as a rectangle.
        """
        mapped = register_map(block_m=32, block_n=64)
        self.assertEqual(mapped.output_tiles, 4)
        self.assertEqual(mapped.resident, 0)
        self.assertEqual(mapped.spilled, 4)
        self.assertIsNotNone(mapped.scratch)

    def test_each_token_group_owns_a_scratch_tile_where_the_file_affords_one(self) -> None:
        """Asserts a spilling block takes one scratch tile per token group until the file runs out.

        A group holding its own tile reloads while the group ahead of it still folds, which is what lets the
        reload issue at a step's head. A block too tall for that falls back to one shared tile.
        """
        for block_m, groups, tiles in ((32, 2, 2), (48, 3, 3), (64, 4, 2), (80, 5, 1)):
            with self.subTest(block_m=block_m):
                mapped = register_map(block_m=block_m, block_n=64 if block_m == 32 else 32)
                self.assertEqual(mapped.token_groups, groups)
                self.assertEqual(mapped.scratch_tiles, tiles)
                self.assertEqual(mapped.reload_hoisted, groups == tiles)

    def test_a_block_the_registers_hold_needs_no_scratch(self) -> None:
        """Asserts a block keeping every output tile resident buys no scratch tile it would never read."""
        mapped = register_map(block_m=32, block_n=32)
        self.assertEqual(mapped.spilled, 0)
        self.assertIsNone(mapped.scratch)

    def test_a_block_with_no_room_for_the_scratch_tile_is_refused(self) -> None:
        """Asserts a block whose staging fills the file exactly is refused before it spills past the file.

        The staging alone fits, so the check covering a block too wide to stage passes it, and the spill it then
        takes wants a scratch tile four registers past the file.
        """
        with self.assertRaises(LayoutError):
            register_map(block_m=96, block_n=16)

    def test_a_block_whose_own_staging_overruns_the_file_is_refused(self) -> None:
        """Asserts a block needing more activation tiles than the file stages raises rather than spilling."""
        with self.assertRaises(LayoutError):
            register_map(block_m=128, block_n=16)

    def test_a_block_that_is_not_whole_tiles_is_refused(self) -> None:
        """Asserts an extent the systolic array cannot cover raises."""
        for block_m, block_n in ((24, 16), (32, 24), (0, 16), (32, 0)):
            with self.subTest(block_m=block_m, block_n=block_n):
                with self.assertRaises(LayoutError):
                    register_map(block_m=block_m, block_n=block_n)


def vmem_row(word: int) -> int | None:
    """Returns the VMEM row one bundle addresses, or None when it addresses none.

    Args:
        word: The bundle word to read.

    Returns:
        The row, taken from the descriptor's address field for a DMA bundle and from the X slot's otherwise.
    """
    if (word >> (asm.BUNDLE_BITS - 1)) & 1:
        return ((word >> asm._D_SHIFT) >> 20) & ((1 << 14) - 1)
    slot = (word >> asm._X_SHIFT) & ((1 << asm.X_SLOT_W) - 1)
    return None if not slot else (slot >> 13) & ((1 << 13) - 1)


def renaming_onto(derived: tuple[int, ...], hand: tuple[int, ...]) -> dict[int, int] | None:
    """Returns the register renaming carrying one bundle stream onto another, or None when none does.

    The emitter lays its registers and its VMEM out from `layout` while the hand-written builder names both
    itself, so the two agree on every slot a bundle occupies and on what it names together, but not on the numbers.
    One consistent renaming of the registers and one of the rows is what says the difference is only those names.

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
    rows: dict[int, int] = {}
    for left, right in zip(derived, hand):
        left_row, right_row = vmem_row(left), vmem_row(right)
        if (left_row is None) != (right_row is None):
            return None
        if left_row is not None and rows.setdefault(left_row, right_row) != right_row:
            return None
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


if __name__ == "__main__":
    unittest.main()
