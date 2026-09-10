"""Contains tests for the GEMM emitter, holding it to the hand-placed image it derives its schedule from.

The emitter exists to replace a hand-tuned bundle stream with one placed from the ISA table, so the only claim worth
testing is that the two agree word for word.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as the package ships its modules beside the tests rather
# than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

import asm  # noqa: E402
import bf16  # noqa: E402
import emit  # noqa: E402
import kernels  # noqa: E402
from compiler_tests.test_layout import renaming_onto  # noqa: E402
from layout import register_map  # noqa: E402


class EmitterConformanceTest(unittest.TestCase):
    """Holds the emitted GEMM to the hand-placed one."""

    def setUp(self) -> None:
        self.hand = tuple(
            kernels.build_looped_gemm(bf16._STEP_TILES, bf16._TOKEN_GROUPS, activate=False).image.assemble())
        self.derived, _ = emit.looped_gemm_stream(
            column_tiles=bf16._STEP_TILES, groups=bf16._TOKEN_GROUPS, loop_blocks=bf16._LOOP_BLOCKS)

    def test_placement_matches_the_hand_written_image(self) -> None:
        """Asserts the emitter places its slots at the same offsets the hand-written kernel does."""
        self.assertEqual(len(self.derived), len(self.hand))

    def test_scheduling_reproduces_the_hand_written_image(self) -> None:
        """Asserts the emitted stream assembles to the same words once the scheduler fills its delays.

        The emitter places slots and leaves every delay field zero, so the scheduler deriving the same delays is what
        says the placement carries the same dependence distances the hand-tuned one does.
        """
        self.assertIsNotNone(renaming_onto(tuple(asm.schedule(list(self.derived))), self.hand))


class EmitterSerialisationTest(unittest.TestCase):
    """Covers the emitted stream as the image bytes a compiler backend writes out."""

    def test_the_emitted_stream_serialises_to_the_hand_written_image(self) -> None:
        """Asserts the emitted stream, wrapped in a kernel, writes the bytes the hand-written builder does."""
        for fuse_groups in (False, True):
            for column_tiles in (1, 3, 6):
                for groups in (1, 3):
                    for loop_blocks in (2, 4):
                        with self.subTest(fuse_groups=fuse_groups, column_tiles=column_tiles, groups=groups,
                                          loop_blocks=loop_blocks):
                            hand = kernels.build_looped_gemm(
                                column_tiles, groups, activate=False, fuse_groups=fuse_groups,
                                loop_blocks=loop_blocks).image.assemble(pad_to=4)
                            derived = asm.AsmKernel(
                                name="looped_gemm",
                                bundles=emit.looped_gemm_stream(
                                    column_tiles=column_tiles, groups=groups, loop_blocks=loop_blocks,
                                    fuse_groups=fuse_groups)[0]).assemble(pad_to=4)
                            self.assertIsNotNone(renaming_onto(derived, hand))


class EmitterRegisterBudgetTest(unittest.TestCase):
    """Holds the emitter's register use to the budget `layout.register_map` derives for the same block."""

    def test_the_emitted_image_touches_the_registers_the_map_lays_out(self) -> None:
        """Asserts the emitted GEMM names every register the map lays out for its block, and no other.

        The emitter takes its bases from the map, so the two agree on which registers a block uses rather than only
        on how many. A base drifting from the map would leave a register named outside its region.
        """
        words, _ = emit.looped_gemm_stream(column_tiles=bf16._STEP_TILES, groups=bf16._TOKEN_GROUPS,
                                           loop_blocks=bf16._LOOP_BLOCKS)
        touched: set[int] = set()
        for word in words:
            decoded = asm._decode(word)
            touched |= set(decoded["reads"]) | {register for register, _ in decoded["writes"]}
        mapped = register_map(block_m=bf16._BLOCK_TOKENS, block_n=bf16._TILE)
        per_tile = mapped.token_groups * bf16._TILE_VREGS
        laid_out: set[int] = set()
        for base, width in ((mapped.accumulator, mapped.resident * per_tile), (mapped.activation, per_tile),
                            (mapped.weight, bf16._TILE_VREGS), (mapped.result, bf16._TILE_VREGS)):
            laid_out |= set(range(base, base + width))
        self.assertEqual(touched, laid_out)


class EmitterHandoverTest(unittest.TestCase):
    """Holds what the emitter tells a host to stage against what the image it placed actually moves."""

    def test_the_staging_drains_the_rows_the_image_stores(self) -> None:
        """Asserts the drain rows a host is told to expect are the rows the image's store descriptors move.

        A host stages and reduces from these figures, so one that drifts from the image leaves the two halves
        describing different launches with nothing between them to disagree.
        """
        for counts in (dict(column_tiles=3, groups=3, loop_blocks=2),
                       dict(column_tiles=4, groups=3, loop_blocks=2, block_n=32),
                       dict(column_tiles=3, groups=3, loop_blocks=2, block_m=16),
                       dict(column_tiles=3, groups=3, loop_blocks=2, fuse_groups=True)):
            with self.subTest(**counts):
                words, staging = emit.looped_gemm_stream(**counts)
                stored = {_descriptor_rows(word) for word in words if _stores(word)}
                self.assertEqual(stored, {staging.drain_rows})


class EmitterStagingTest(unittest.TestCase):
    """Covers the staging analysis the emitter places its loads from."""

    def test_a_step_places_one_bundle_per_offset(self) -> None:
        """Asserts one GEMM step names no offset twice, since two slots of a kind cannot share a bundle."""
        step = emit.gemm_step(activation_row=0, weight_row=0, close_loop=False)
        for offset, slots in step.items():
            self.assertEqual(len(slots), len(set(slots)), f"offset {offset} names a slot twice")


def _stores(word: int) -> bool:
    """Returns whether one bundle carries a descriptor writing host memory."""
    return bool(asm._decode(word)["dma"]) and bool((word >> asm._D_SHIFT) >> 35 & 1)


def _descriptor_rows(word: int) -> int:
    """Returns the rows one descriptor moves, which it holds one less than."""
    return ((word >> asm._D_SHIFT) >> 4 & 0x7F) + 1


if __name__ == "__main__":
    unittest.main()
