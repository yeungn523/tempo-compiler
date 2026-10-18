"""Contains tests for an attention kernel selecting its scores by position, from the reader through to the packing."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as neither the board package nor the compiler is
# installed.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
"""The repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "board_package" / "minitpu_runtime"))

import numpy as np  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import emit  # noqa: E402
from bf16 import _TILE, as_bf16_words, pack_attention  # noqa: E402
from compiler import ProgramError, compile_launch  # noqa: E402
from compiler.ir import MASK, Affine  # noqa: E402
from compiler_tests.test_compile import planned_attention, read_attention  # noqa: E402


class MaskReaderTest(unittest.TestCase):
    """Covers how the reader reads a select over the scores, and what it refuses."""

    def test_a_causal_select_is_read_as_a_comparison_of_positions(self) -> None:
        """Asserts the query's position is read as its loop plus the tile's row, and the key's as its own."""
        mask, _ = read_attention(causal_attention)[1].mask
        self.assertEqual(mask.predicate, 5)
        self.assertEqual(mask.left, Affine(query=1, row=1))
        self.assertEqual(mask.right, Affine(key=1, column=1))
        self.assertTrue(mask.keeps)
        self.assertEqual(mask.fill, -1e30)

    def test_the_select_is_recorded_as_the_mask_add(self) -> None:
        """Asserts the trip adds a value to the scores in place of the select, which reads the predicate."""
        attended = read_attention(causal_attention)[1]
        names = [operation.name for operation in attended.prologue_chain]
        self.assertIn(MASK, names)
        self.assertNotIn("arith.select", names)

    def test_an_unmasked_kernel_reads_no_mask(self) -> None:
        """Asserts a kernel selecting nothing carries no mask."""
        self.assertIsNone(read_attention()[1].mask)

    def test_a_select_the_image_cannot_run_is_refused(self) -> None:
        """Asserts a fill some score could reach, and a second select, each raise rather than compiling."""
        for kernel, reason in ((shallow_attention, "or less"), (doubly_masked_attention, "selecting 2 times")):
            with self.subTest(kernel=kernel.fn.__name__), self.assertRaises(ProgramError) as raised:
                read_attention(kernel)
            self.assertIn(reason, str(raised.exception))


class MaskTileTest(unittest.TestCase):
    """Holds the tiles a mask evaluates to against the predicate evaluated element by element."""

    def test_every_block_pair_matches_the_predicate(self) -> None:
        """Asserts the causal and the windowed tiles agree with each kernel's own predicate at every position."""
        for kernel, kept in ((causal_attention, lambda query, key: query >= key),
                             (windowed_attention, lambda query, key: query - key < 8)):
            with self.subTest(kernel=kernel.fn.__name__):
                launch = planned_attention(kernel)
                tiles = launch.mask_tiles
                self.assertEqual(tiles.shape, (launch.row_blocks, launch.key_blocks, _TILE, _TILE))
                queries = np.arange(launch.row_blocks * _TILE)[:, None]
                keys = np.arange(launch.key_blocks * _TILE)[None, :]
                expected = np.where(kept(queries, keys), 0.0, -1e30)
                np.testing.assert_array_equal(launch.mask_scores, expected.astype(np.float32))


class MaskLaunchTest(unittest.TestCase):
    """Covers the steps and the image a masked launch carries."""

    def test_each_trip_adds_its_own_mask_tile_before_the_maximum(self) -> None:
        """Asserts the body loads the mask through the key loop's counter and adds it ahead of the row maximum."""
        body = planned_attention(causal_attention).body
        loads = [index for index, step in enumerate(body) if isinstance(step, emit.Load) and step.region == "mask"]
        self.assertEqual(len(loads), 1)
        self.assertTrue(body[loads[0]].walked)
        maximum = next(index for index, step in enumerate(body)
                       if isinstance(step, emit.RowReduce) and step.kind == "vmax")
        self.assertLess(loads[0], maximum)

    def test_the_mask_tiles_ride_an_extra_staging_descriptor(self) -> None:
        """Asserts four mask tiles push a block's staging past one descriptor, and the masked image differs."""
        plain, masked = planned_attention(), planned_attention(causal_attention)

        def staging_descriptors(launch) -> int:
            """Returns the descriptors one image issues outside its key loop."""
            bundles, _ = emit.flash_attention(key_blocks=launch.key_blocks, queries=launch.queries, head=launch.head,
                                              body=launch.body, epilogue=launch.epilogue,
                                              constants=len(launch.constants), row_blocks=launch.row_blocks,
                                              masked=launch.mask is not None)
            return sum("d" in slots for slots in bundles.values())

        self.assertEqual(staging_descriptors(masked), staging_descriptors(plain) + 1)
        self.assertNotEqual(compile_launch(plain)[0], compile_launch(masked)[0])

    def test_the_host_packs_each_block_s_mask_tiles_after_its_constants(self) -> None:
        """Asserts a query block's staging closes with its mask tiles, one per key block in order."""
        launch = planned_attention(causal_attention)
        rng = np.random.default_rng(0)
        query = rng.standard_normal((launch.row_blocks * _TILE, launch.head)).astype(np.float32)
        key = rng.standard_normal((launch.key_blocks * _TILE, launch.head)).astype(np.float32)
        staged, _ = pack_attention(query, key, key, initial_maximum=launch.entering, constants=launch.constants,
                                   masks=launch.mask_tiles)
        block_rows = launch.head + (2 + len(launch.constants) + launch.key_blocks) * _TILE
        blocks = staged.reshape(launch.row_blocks, block_rows, _TILE)
        np.testing.assert_array_equal(blocks[:, -launch.key_blocks * _TILE:],
                                      as_bf16_words(values=launch.mask_tiles).reshape(launch.row_blocks, -1, _TILE))


@triton.jit
def causal_attention(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                     stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64, D: tl.constexpr = 64,
                     BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16, FILL: tl.constexpr = -1e30):
    """Attends as the plain kernel does, a query reading no key after its own position."""
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(qp + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(kp + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            value = tl.load(vp + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], tl.dot(query, tl.trans(key)), FILL)
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, accumulator / running_sum[:, None])


@triton.jit
def windowed_attention(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                       stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64, D: tl.constexpr = 64,
                       BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16):
    """Attends as the plain kernel does, a query reading only keys fewer than eight positions before it or later."""
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(qp + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(kp + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            value = tl.load(vp + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            scores = tl.where(offs_m[:, None] - offs_n[None, :] < 8, tl.dot(query, tl.trans(key)), -1e30)
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, accumulator / running_sum[:, None])


@triton.jit
def shallow_attention(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                      stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64, D: tl.constexpr = 64,
                      BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16):
    """Masks as the causal kernel does, but to a fill of -100, which a score could reach."""
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(qp + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(kp + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            value = tl.load(vp + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], tl.dot(query, tl.trans(key)), -100.0)
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, accumulator / running_sum[:, None])


@triton.jit
def doubly_masked_attention(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                            stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64, D: tl.constexpr = 64,
                            BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16):
    """Masks causally and then by a window, two selects where the image stages one mask tile per block pair."""
    offs_d = tl.arange(0, D)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        query = tl.load(qp + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            key = tl.load(kp + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            value = tl.load(vp + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], tl.dot(query, tl.trans(key)), -1e30)
            scores = tl.where(offs_m[:, None] - offs_n[None, :] < 8, scores, -1e30)
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, accumulator / running_sum[:, None])


if __name__ == "__main__":
    unittest.main()
