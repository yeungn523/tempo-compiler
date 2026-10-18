"""Contains tests for the attention images and the copy filling their KV cache."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as neither the board package nor the code generator
# is installed.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

import numpy as np  # noqa: E402

from bf16 import _TILE, as_bf16_words, decode_mask_tiles, pack_attention_constants  # noqa: E402
from codegen import ProgramError, compile_resident_attention, plan, read_program, signature_of  # noqa: E402
from codegen import asm  # noqa: E402
from codegen import emit  # noqa: E402
from codegen.layout import flash_register_map  # noqa: E402
from tests.test_compile import planned_attention  # noqa: E402
from tests.test_emit import _e_opcode  # noqa: E402
from tests.test_mask import causal_attention  # noqa: E402
import gpt2_triton  # noqa: E402


_FUSED_TILES: int = 3 * 768 // _TILE
"""Feature tiles one token of GPT-2 small's fused QKV tensor holds."""

_VTXIN: int = 13
"""E-slot opcode of vtxin, which stages one VREG into the transpose array."""


class ResidentAttentionImageTest(unittest.TestCase):
    """Covers the image a masked launch compiles to against the resident layout."""

    def test_the_image_fits_and_copies_every_tile_it_reads_and_writes(self) -> None:
        """Asserts a query half copies its constants, queries and output, and each key half its keys and values."""
        launch = planned_attention(causal_attention)
        image = compile_resident_attention(launch, heads=12, fused_tiles=_FUSED_TILES)
        self.assertLessEqual(len(image), asm.IRAM_DEPTH * asm.BUNDLE_BYTES)
        constants, queries, output, keys_and_values = 2 + launch.key_blocks, 4, 4, 2 * (4 + 4)
        self.assertEqual(sum((word >> (asm.BUNDLE_BITS - 1)) & 1 for word in resident_words(launch)),
                         constants + queries + output + keys_and_values)

    def test_each_key_tile_is_transposed_in_registers(self) -> None:
        """Asserts every key tile of both key halves passes through the transpose array on its way to the array."""
        launch = planned_attention(causal_attention)
        image, body, epilogue = (transposes(words) for words in (resident_words(launch), steps(launch.body),
                                                                  steps(launch.epilogue)))
        self.assertEqual(image - 2 * body - epilogue, 2 * 4 * 4)

    def test_a_launch_the_image_cannot_run_is_refused(self) -> None:
        """Asserts an unmasked launch, and a sequence of other than whole token blocks, each raise."""
        with self.assertRaises(ProgramError):
            compile_resident_attention(planned_attention(), heads=12, fused_tiles=_FUSED_TILES)
        with self.assertRaises(ValueError):
            emit.resident_attention_stream(key_blocks=3, head=64, body=(), epilogue=(), constants=0, heads=12,
                                           fused_tiles=_FUSED_TILES)


class DecodeAttentionTest(unittest.TestCase):
    """Covers the decode image, which attends one query block over a KV cache bound on its own."""

    def test_a_decode_kernel_plans_one_query_block_masked_past_its_length(self) -> None:
        """Asserts the folded query loop reads as one block, and every query row keeps exactly the first keys."""
        launch = planned_decode(keys=64, length=48)
        self.assertEqual((launch.row_blocks, launch.key_blocks), (1, 4))
        kept = launch.mask_scores == 0
        self.assertTrue(kept[:, :48].all())
        self.assertFalse(kept[:, 48:].any())

    def test_the_board_writes_the_mask_the_kernel_states(self) -> None:
        """Asserts the tiles a board writes for a step equal the ones the compiler derives from the kernel's select."""
        for length in (1, 33, 48, 64):
            with self.subTest(length=length):
                np.testing.assert_array_equal(decode_mask_tiles(keys=64, length=length),
                                              planned_decode(keys=64, length=length).mask_tiles)

    def test_the_decode_image_fits_and_reads_the_cache_as_its_own_argument(self) -> None:
        """Asserts the image compiles against a cache of a key and a value per feature, and refuses another."""
        launch = planned_decode(keys=64, length=48)
        image = compile_resident_attention(launch, heads=12, fused_tiles=_FUSED_TILES,
                                           cache_tiles=2 * _FUSED_TILES // 3)
        self.assertLessEqual(len(image), asm.IRAM_DEPTH * asm.BUNDLE_BYTES)
        with self.assertRaises(ValueError):
            compile_resident_attention(launch, heads=12, fused_tiles=_FUSED_TILES, cache_tiles=_FUSED_TILES)


class CopyImageTest(unittest.TestCase):
    """Covers the strided copy that fills the KV cache."""

    def test_a_copy_moves_its_rows_in_descriptors_of_at_most_128(self) -> None:
        """Asserts a token block's keys and values take 24 load and store pairs, and one token's take one."""
        for rows, stride, descriptors in ((3072, 32, 2 * 24), (96, 1024, 2)):
            with self.subTest(rows=rows, stride=stride):
                words = emit.strided_copy_stream(rows=rows, stride=stride)
                self.assertEqual(sum((word >> (asm.BUNDLE_BITS - 1)) & 1 for word in words), descriptors)

    def test_a_copy_the_machine_cannot_make_is_refused(self) -> None:
        """Asserts an empty copy and a stride of part of a row each raise."""
        for rows, stride in ((0, 32), (96, 48)):
            with self.subTest(rows=rows, stride=stride), self.assertRaises(ValueError):
                emit.strided_copy_stream(rows=rows, stride=stride)


class AttentionConstantsTest(unittest.TestCase):
    """Covers the constants block a host stages for each query block."""

    def test_each_query_block_carries_its_entering_tiles_then_its_mask_tiles(self) -> None:
        """Asserts a block's rows open with the initial maximum and the zeros and close with its own mask tiles."""
        launch = planned_attention(causal_attention)
        packed = pack_attention_constants(initial_maximum=launch.entering, constants=launch.constants,
                                          masks=launch.mask_tiles)
        blocks = packed.reshape(launch.row_blocks, 2 + launch.key_blocks, _TILE, _TILE)
        initial = as_bf16_words(values=np.full((_TILE, _TILE), launch.entering, dtype=np.float32))
        np.testing.assert_array_equal(blocks[:, 0], np.broadcast_to(initial, blocks[:, 0].shape))
        self.assertFalse(blocks[:, 1].any())
        np.testing.assert_array_equal(blocks[:, 2:], as_bf16_words(values=launch.mask_tiles))


def planned_decode(*, keys: int, length: int):
    """Returns the launch GPT-2's decode attention kernel plans to over a cache of the given keys and length."""
    kernel = gpt2_triton.gpt2_decode_attention
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands={}),
                           constexprs=declared | {"N": keys, "LENGTH": length})
    return plan(matmuls, strides={}).launches[0]


def steps(body) -> tuple[int, ...]:
    """Returns the bundle words one body of steps emits on its own."""
    bundles: dict[int, dict[str, int]] = {}

    def put(*, offset: int, **slots: int) -> None:
        """Records the slot words one bundle carries."""
        bundles.setdefault(offset, {}).update(slots)

    emit.emit_steps(put, asm.AsmBuilder(), 0, body=body, registers=flash_register_map(queries=_TILE, head=64),
                    rows=emit._flash_vmem_map(head=64, masks=4), turnaround=8)
    return emit._flatten(bundles)


def transposes(words: tuple[int, ...]) -> int:
    """Returns how many VREGs the words stage into the transpose array."""
    return sum(_e_opcode(word) == _VTXIN for word in words)


def resident_words(launch) -> tuple[int, ...]:
    """Returns the bundle words the resident image emits for one launch, before placement."""
    return emit.resident_attention_stream(key_blocks=launch.key_blocks, head=launch.head, body=launch.body,
                                          epilogue=launch.epilogue, constants=len(launch.constants), heads=12,
                                          fused_tiles=_FUSED_TILES)


if __name__ == "__main__":
    unittest.main()
