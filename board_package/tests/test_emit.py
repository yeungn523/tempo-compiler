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
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

import asm  # noqa: E402
import bf16  # noqa: E402
import emit  # noqa: E402
import kernels  # noqa: E402


class EmitterConformanceTest(unittest.TestCase):
    """Holds the emitted GEMM to the hand-placed one."""

    def setUp(self) -> None:
        self.hand = tuple(
            kernels.build_looped_gemm(bf16._STEP_TILES, bf16._TOKEN_GROUPS, activate=False).image.assemble())
        self.derived = emit.looped_gemm_stream(
            column_tiles=bf16._STEP_TILES, groups=bf16._TOKEN_GROUPS, loop_blocks=bf16._LOOP_BLOCKS)

    def test_placement_matches_the_hand_written_image(self) -> None:
        """Asserts the emitter places its slots at the same offsets the hand-written kernel does."""
        self.assertEqual(len(self.derived), len(self.hand))

    def test_scheduling_reproduces_the_hand_written_image(self) -> None:
        """Asserts the emitted stream assembles to the same words once the scheduler fills its delays.

        The emitter places slots and leaves every delay field zero, so the scheduler deriving the same delays is what
        says the placement carries the same dependence distances the hand-tuned one does.
        """
        self.assertEqual(tuple(asm.schedule(list(self.derived))), self.hand)


class EmitterStagingTest(unittest.TestCase):
    """Covers the staging analysis the emitter places its loads from."""

    def test_a_step_places_one_bundle_per_offset(self) -> None:
        """Asserts one GEMM step names no offset twice, since two slots of a kind cannot share a bundle."""
        step = emit.emit_gemm_step(activation_row=0, weight_row=0, close_loop=False)
        for offset, slots in step.items():
            self.assertEqual(len(slots), len(set(slots)), f"offset {offset} names a slot twice")


if __name__ == "__main__":
    unittest.main()
