"""Check that a test's operands can observe the confusions a tiled GEMM can make."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from gpt2_kernels import _BLOCK_DEPTH, _on_bf16_grid, assert_operands_discriminating  # noqa: E402

TOKENS, DEPTH, COLUMNS = 32, 192, 16


def varying() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        _on_bf16_grid(values=rng.standard_normal((TOKENS, DEPTH)).astype(np.float32)),
        _on_bf16_grid(values=rng.standard_normal((DEPTH, COLUMNS)).astype(np.float32)),
    )


class OperandDiscriminationTest(unittest.TestCase):
    """A test is only evidence for the axes its operands vary along.

    The operand set that stood as the strongest board evidence for weeks --
    `arange % 2` against weights broadcast along the contraction -- passed at
    rtol=0 atol=0 while the kernel was dropping every result the MXU output FIFO
    overflowed. It could not have failed: identical token rows and weights
    constant along the contraction make every 32-deep block contribute the same
    value, so a dropped block's stale result sums to the same answer.
    """

    def test_varying_operands_are_accepted(self) -> None:
        left, right = varying()
        assert_operands_discriminating(activations=left, weights=right)

    def test_identical_token_rows_are_rejected(self) -> None:
        _, right = varying()
        left = np.tile(np.arange(DEPTH, dtype=np.float32), (TOKENS, 1))
        with self.assertRaisesRegex(ValueError, "tokens"):
            assert_operands_discriminating(activations=left, weights=right)

    def test_weights_constant_along_the_contraction_are_rejected(self) -> None:
        left, _ = varying()
        right = np.broadcast_to(
            np.arange(1, COLUMNS + 1, dtype=np.float32), (DEPTH, COLUMNS)
        ).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "contraction blocks"):
            assert_operands_discriminating(activations=left, weights=right)

    def test_identical_output_columns_are_rejected(self) -> None:
        left, _ = varying()
        column = np.arange(DEPTH, dtype=np.float32).reshape(DEPTH, 1)
        right = np.repeat(column, COLUMNS, axis=1)
        with self.assertRaisesRegex(ValueError, "output columns"):
            assert_operands_discriminating(activations=left, weights=right)

    def test_the_operands_that_hid_the_defect_are_rejected(self) -> None:
        left = np.arange(TOKENS * DEPTH, dtype=np.float32).reshape(TOKENS, DEPTH) % 2
        scales = (2.0 ** -(np.arange(COLUMNS) % 4)) / 16.0
        right = np.broadcast_to(scales, shape=(DEPTH, COLUMNS)).astype(np.float32)
        with self.assertRaises(ValueError):
            assert_operands_discriminating(activations=left, weights=right)

    def test_a_whole_number_of_blocks_is_assumed(self) -> None:
        self.assertEqual(DEPTH % _BLOCK_DEPTH, 0)


if __name__ == "__main__":
    unittest.main()
