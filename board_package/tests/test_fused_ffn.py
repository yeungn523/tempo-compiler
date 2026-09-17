"""Contains tests for the operands the fused feed-forward layer is staged from."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the package imports below resolve, as the package ships its modules beside
# the tests rather than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _TOKEN_GROUPS,
    as_bf16_words,
    fused_ffn_reference,
    pack_fused_ffn,
)


class FusedFeedForwardPackingTest(unittest.TestCase):
    """Verifies that the host lays the operands out where the kernel's loads read them."""

    def test_the_input_covers_the_rows_its_descriptor_stages(self) -> None:
        """Asserts the packed input is exactly the rows one descriptor moves for the contraction it covers."""
        for steps in (1, 2, 3):
            with self.subTest(steps=steps):
                staged, _ = _operands(steps=steps, chunks=2)
                self.assertEqual(staged.size, steps * _TOKEN_GROUPS * _TILE * _TILE)

    def test_each_chunk_covers_the_rows_its_descriptor_stages(self) -> None:
        """Asserts one chunk of weights is the up projection's tiles followed by the down projection's tile."""
        for steps, chunks in ((2, 2), (3, 4)):
            with self.subTest(steps=steps, chunks=chunks):
                _, weights = _operands(steps=steps, chunks=chunks)
                padded = (chunks * (steps * _TILE + _TILE) + steps * _TILE) * _TILE
                self.assertEqual(weights.size, padded)

    def test_the_input_lands_where_the_loads_read_it(self) -> None:
        """Asserts every token and contraction position sits in the VMEM row and lane its tile's load covers."""
        activations, up_weight, down_weight = _arrays(steps=2, chunks=2)
        staged, _ = pack_fused_ffn(activations, up_weight, down_weight)
        expected = as_bf16_words(values=activations)
        for token in range(_BLOCK_TOKENS):
            for position in range(activations.shape[1]):
                tile = (position // _TILE) * _TOKEN_GROUPS + token // _TILE
                word = (tile * _TILE + token % _TILE) * _TILE + position % _TILE
                self.assertEqual(staged[word], expected[token, position])

    def test_each_chunk_holds_the_hidden_columns_it_contracts(self) -> None:
        """Asserts a chunk carries the up projection columns and the down projection rows that same chunk uses."""
        activations, up_weight, down_weight = _arrays(steps=2, chunks=2)
        _, weights = pack_fused_ffn(activations, up_weight, down_weight)
        depth, hidden = up_weight.shape
        span = (depth + _TILE) * _TILE
        for chunk in range(hidden // _TILE):
            first = chunk * _TILE
            up = weights[chunk * span:chunk * span + depth * _TILE]
            down = weights[chunk * span + depth * _TILE:(chunk + 1) * span]
            self.assertTrue(np.array_equal(
                up.reshape(depth, _TILE), as_bf16_words(values=up_weight[:, first:first + _TILE])))
            self.assertTrue(np.array_equal(
                down.reshape(_TILE, _TILE), as_bf16_words(values=down_weight[first:first + _TILE])))

    def test_a_shape_the_layer_does_not_cover_is_refused(self) -> None:
        """Asserts an input, a hidden width or an output the fused layer cannot hold raises."""
        activations, up_weight, down_weight = _arrays(steps=2, chunks=2)
        cases = (
            (activations[:_TILE], up_weight, down_weight),
            (activations, up_weight[:, :_TILE - 1], down_weight),
            (activations, up_weight, down_weight[:, :_TILE - 1]),
            (activations, up_weight, down_weight[:_TILE]),
        )
        for case in cases:
            with self.subTest(shapes=[operand.shape for operand in case]), self.assertRaises(ValueError):
                pack_fused_ffn(*case)

    def test_the_reference_activates_between_the_projections(self) -> None:
        """Asserts the host reference applies the activation to the intermediate rather than to either product."""
        activations, up_weight, down_weight = _arrays(steps=2, chunks=2)
        hidden = activations.astype(np.float64) @ up_weight.astype(np.float64)
        plain = hidden @ down_weight.astype(np.float64)
        layer = fused_ffn_reference(activations, up_weight, down_weight)
        self.assertEqual(layer.shape, (_BLOCK_TOKENS, _TILE))
        self.assertFalse(np.allclose(layer, plain))


def _arrays(*, steps: int, chunks: int):
    """Returns an input and both projections at the extents the given loop counts cover."""
    rng = np.random.default_rng(0)
    depth, hidden = steps * _TILE, chunks * _TILE
    return (rng.standard_normal((_BLOCK_TOKENS, depth), dtype=np.float32),
            rng.standard_normal((depth, hidden), dtype=np.float32),
            rng.standard_normal((hidden, _TILE), dtype=np.float32))


def _operands(*, steps: int, chunks: int):
    """Returns the packed operands at the extents the given loop counts cover."""
    return pack_fused_ffn(*_arrays(steps=steps, chunks=chunks))


if __name__ == "__main__":
    unittest.main()
