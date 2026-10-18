"""Contains tests for the operand discrimination guard used by the tiled GEMM kernels.

Identical token rows, or weights constant along the contraction, make every 32-deep block contribute the same value,
so a dropped block still sums to the right answer and an exact comparison passes over a broken kernel.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Extends the import path before the kernel imports below resolve, as the package ships its modules beside the tests
# rather than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from bf16 import _BLOCK_DEPTH, _on_bf16_grid, assert_operands_discriminating  # noqa: E402

_TOKENS: int = 32
"""The number of token rows in the test activations."""

_DEPTH: int = 192
"""The length of the contraction axis shared by the test operands."""

_COLUMNS: int = 16
"""The number of output columns in the test weights."""


class OperandDiscriminationTest(unittest.TestCase):
    """Verifies that the guard accepts discriminating operands and rejects degenerate ones."""

    def test_varying_operands_are_accepted(self) -> None:
        """Verifies that operands varying along every axis pass the guard."""
        activations, weights = _varying_operands()
        assert_operands_discriminating(activations=activations, weights=weights)

    def test_identical_token_rows_are_rejected(self) -> None:
        """Verifies that activations repeating one token row are rejected."""
        _, weights = _varying_operands()
        activations = np.tile(np.arange(_DEPTH, dtype=np.float32), (_TOKENS, 1))
        with self.assertRaisesRegex(ValueError, "tokens"):
            assert_operands_discriminating(activations=activations, weights=weights)

    def test_weights_constant_along_the_contraction_are_rejected(self) -> None:
        """Verifies that weights broadcast along the contraction axis are rejected."""
        activations, _ = _varying_operands()
        weights = np.broadcast_to(
            np.arange(1, _COLUMNS + 1, dtype=np.float32), (_DEPTH, _COLUMNS)
        ).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "contraction blocks"):
            assert_operands_discriminating(activations=activations, weights=weights)

    def test_identical_output_columns_are_rejected(self) -> None:
        """Verifies that weights repeating one output column are rejected."""
        activations, _ = _varying_operands()
        column = np.arange(_DEPTH, dtype=np.float32).reshape(_DEPTH, 1)
        weights = np.repeat(column, _COLUMNS, axis=1)
        with self.assertRaisesRegex(ValueError, "output columns"):
            assert_operands_discriminating(activations=activations, weights=weights)

    def test_two_contraction_blocks_are_accepted(self) -> None:
        """Verifies that operands are judged on their own content when only two contraction blocks exist.

        Two blocks admit two orderings, so a permutation drawn at random is the identity half the time. The guard
        then compares an operand against itself, which says nothing about whether a reordering would show.
        """
        activations, weights = _varying_operands(depth=2 * _BLOCK_DEPTH)
        assert_operands_discriminating(activations=activations, weights=weights)

def _varying_operands(depth: int = _DEPTH) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Returns an activation and weight pair that varies along every axis the guard inspects.

    Args:
        depth: The contraction the pair shares.
    """
    generator = np.random.default_rng(0)
    return (
        _on_bf16_grid(values=generator.standard_normal((_TOKENS, depth)).astype(np.float32)),
        _on_bf16_grid(values=generator.standard_normal((depth, _COLUMNS)).astype(np.float32)),
    )


if __name__ == "__main__":
    unittest.main()
