"""Contains tests for the register file map a block's tiling derives.

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

from isa_model import PROFILE  # noqa: E402
from layout import LayoutError, register_map  # noqa: E402


class RegisterMapTest(unittest.TestCase):
    """Covers the register map across the block shapes a kernel may ask for."""

    def test_the_regions_are_disjoint_and_fit_the_file(self) -> None:
        """Asserts no two regions overlap and the map never hands out a register the file lacks."""
        entries = PROFILE["vreg_entry_count"]
        for block_m in (16, 32, 48, 64):
            for block_n in (16, 32, 64):
                with self.subTest(block_m=block_m, block_n=block_n):
                    mapped = register_map(block_m=block_m, block_n=block_n)
                    per_tile = mapped.token_groups * 4
                    spans = [(mapped.accumulator, mapped.resident * per_tile),
                             (mapped.activation, per_tile), (mapped.weight, 4), (mapped.result, 4)]
                    occupied: set[int] = set()
                    for base, width in spans:
                        registers = set(range(base, base + width))
                        self.assertFalse(registers & occupied, "regions overlap")
                        occupied |= registers
                    self.assertLessEqual(mapped.used, entries)
                    self.assertLessEqual(max(occupied), entries - 1)

    def test_a_block_wider_than_the_file_spills_rather_than_refusing(self) -> None:
        """Asserts output tiles past what the registers hold accumulate in VMEM instead of raising."""
        mapped = register_map(block_m=32, block_n=64)
        self.assertEqual(mapped.output_tiles, 4)
        self.assertEqual(mapped.resident, 2)
        self.assertEqual(mapped.spilled, 2)

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


if __name__ == "__main__":
    unittest.main()
