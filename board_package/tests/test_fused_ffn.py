"""Contains tests for the image the fused feed-forward layer assembles to."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the kernel imports below resolve, as the package ships its modules beside the
# tests rather than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))

import asm  # noqa: E402
from asm import E_SLOT_W, M_SLOT_W, X_SLOT_W, _E_SHIFT, _M_SHIFT, _X_SHIFT  # noqa: E402
import numpy as np  # noqa: E402

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _TILE_VREGS,
    _TOKEN_GROUPS,
    as_bf16_words,
    fused_ffn_reference,
    pack_fused_ffn,
)
from kernels import _RESIDENT_TILES, build_fused_ffn  # noqa: E402

_GELU: int = asm._OPCODES["vgelu"]
"""E-slot opcode of the activation, which the counts below pick out of the assembled image."""

_TILES: int = _TOKEN_GROUPS * _TILE_VREGS
"""Registers one token block's hidden tiles occupy, being one activation each."""


class FusedFeedForwardTest(unittest.TestCase):
    """Verifies the image the fused feed-forward layer assembles to."""

    def test_the_activation_runs_once_per_hidden_chunk(self) -> None:
        """Asserts the chunk is activated once however many output tiles read it.

        The activation stands ahead of the output sweep rather than inside it, one riding the launch bundles in
        that loop being one that rewrites the chunk again on every tile.
        """
        for output_tiles in (1, 4):
            with self.subTest(output_tiles=output_tiles):
                counts = _counts(build_fused_ffn(chunks=2, output_tiles=output_tiles))
                self.assertEqual(counts["vgelu"], _TILES)

    def test_the_intermediate_never_reaches_vmem(self) -> None:
        """Asserts no store names a register the hidden chunk occupies, it passing between the products in place."""
        hidden = {vreg for tile in _RESIDENT_TILES for vreg in range(tile, tile + _TILE_VREGS)}
        self.assertTrue(hidden.isdisjoint(_stored_registers(build_fused_ffn(chunks=2, output_tiles=4))))

    def test_the_output_sweep_rides_a_loop(self) -> None:
        """Asserts a wider output grows the trip count rather than the stores the image carries."""
        narrow, wide = (_counts(build_fused_ffn(chunks=2, output_tiles=tiles)) for tiles in (1, 8))
        self.assertEqual(narrow["vst"], wide["vst"])
        self.assertEqual(narrow["vmatpush"], wide["vmatpush"])

    def test_the_image_carries_one_step_body_and_the_down_projection(self) -> None:
        """Asserts the contraction rides a loop's trip count, the image holding two rounds of pushes at any depth.

        A step body per contraction tile would put GPT-2's 48 of them past the instruction memory, so the image
        staying flat is what the AGU walking the step is for.
        """
        for steps in (2, 3, 48):
            with self.subTest(steps=steps):
                counts = _counts(build_fused_ffn(chunks=2, steps=steps))
                self.assertEqual(counts["vmatpush"], 2 * _TILES)
                self.assertEqual(counts["vmatpop"], counts["vmatpush"])

    def test_a_deeper_contraction_costs_no_further_step_bodies(self) -> None:
        """Asserts a contraction tile costs less than its own pushes, so the step body is not in the image twice.

        What a deeper contraction does add is the descriptors that stage it, one step body emitted per tile
        being what put GPT-2's 48 of them past the instruction memory the loader fills.
        """
        shallow = len(build_fused_ffn(chunks=2, steps=2).image.assemble())
        deep = len(build_fused_ffn(chunks=2, steps=48).image.assemble())
        self.assertLess((deep - shallow) / (48 - 2), _TILES)

    def test_the_image_is_the_same_size_at_every_chunk_count(self) -> None:
        """Asserts the hidden width rides the loop's trip count rather than growing the image."""
        sizes = {len(build_fused_ffn(chunks=chunks).image.assemble()) for chunks in (1, 2, 8)}
        self.assertEqual(len(sizes), 1)

    def test_an_empty_extent_is_refused(self) -> None:
        """Asserts a layer covering no hidden chunk or no contraction tile raises."""
        for chunks, steps in ((0, 2), (2, 0)):
            with self.subTest(chunks=chunks, steps=steps), self.assertRaises(ValueError):
                build_fused_ffn(chunks=chunks, steps=steps)

    def test_a_contraction_overrunning_the_vmem_slot_is_refused(self) -> None:
        """Asserts a contraction whose staging does not fit one VMEM slot raises rather than wrapping inside it."""
        with self.assertRaisesRegex(ValueError, "VMEM"):
            build_fused_ffn(chunks=1, steps=1024)


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
                self.assertEqual(weights.size, chunks * (steps * _TILE + _TILE) * _TILE)

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


def _stored_registers(kernel) -> set[int]:
    """Returns every VREG the kernel's image stores out to VMEM."""
    found = set()
    for word in kernel.image.assemble():
        if (word >> 127) & 1:
            continue
        x = (word >> _X_SHIFT) & ((1 << X_SLOT_W) - 1)
        if x and (x >> 31) & 1:
            found.add((x >> 26) & 0x1F)
    return found


def _counts(kernel) -> dict[str, int]:
    """Returns how many of each operation the kernel's assembled image issues."""
    found = dict.fromkeys(("vgelu", "riding", "vst", "dma_store", "vmatpush", "vmatpop"), 0)
    for word in kernel.image.assemble():
        if (word >> 127) & 1:
            found["dma_store"] += (word >> 126) & 1
            continue
        e = (word >> _E_SHIFT) & ((1 << E_SLOT_W) - 1)
        m = (word >> _M_SHIFT) & ((1 << M_SLOT_W) - 1)
        x = (word >> _X_SHIFT) & ((1 << X_SLOT_W) - 1)
        m_op = (m >> 5) & 0x3 if m else 0
        activation = bool(e) and (e >> 15) & 0xF == _GELU
        found["vgelu"] += activation
        found["riding"] += activation and m_op in (1, 2)
        found["vmatpush"] += m_op == 2
        found["vmatpop"] += m_op == 3
        found["vst"] += bool(x) and (x >> 31) & 1
    return found


if __name__ == "__main__":
    unittest.main()
