"""Contains tests for tile size relaxation and boundary enforcement.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from dataclasses import replace

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "board_package" / "minitpu_runtime"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from compiler import ProgramError, compile_launch, plan, read_program, signature_of  # noqa: E402
from compiler.ir import MatMul  # noqa: E402
from compiler.planner import _machine_tile  # noqa: E402


@triton.jit
def gemm_half_k(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 32):
    """Multiplies with a reduction step of 32, which the planner pads to the 64 rows one unrolled group covers."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_quarter_k(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                   BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 16):
    """Multiplies with a reduction step of 16, which the planner pads to 64."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_excessive_n(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     M: tl.constexpr = 64, N: tl.constexpr = 256, K: tl.constexpr = 192,
                     BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 128, BLOCK_K: tl.constexpr = 64):
    """Multiplies with BLOCK_N=128 which exceeds IRAM bundle capacity."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_folded_m(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  M: tl.constexpr = 64, N: tl.constexpr = 64, K: tl.constexpr = 192,
                  BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 32, BLOCK_K: tl.constexpr = 64):
    """Multiplies with M == BLOCK_M == 64, where Triton folds the single-iteration m loop away."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_excessive_m(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     M: tl.constexpr = 256, N: tl.constexpr = 64, K: tl.constexpr = 192,
                     BLOCK_M: tl.constexpr = 128, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies with BLOCK_M=128 which exceeds vector register count."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_grid_m(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                M: tl.constexpr = 256, N: tl.constexpr = 64, K: tl.constexpr = 192,
                BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies with the token axis on the grid, where the module states no extent along m at all."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(0, N, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            accumulator += tl.dot(tl.load(a), tl.load(b))
            a += BLOCK_K * stride_ak
            b += BLOCK_K * stride_bk
        tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_unplaceable_n(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       M: tl.constexpr = 256, N: tl.constexpr = 256, K: tl.constexpr = 192,
                       BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 256, BLOCK_K: tl.constexpr = 64):
    """Multiplies with BLOCK_N=256, whose unrolled image overflows IRAM under either placement."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(a), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


_OPERANDS = {"ap": "*bf16", "wp": "*bf16", "op": "*fp32"}
_STRIDE_NAMES = ("stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn")


def _compile_helper(kernel, *, dims=None):
    constexprs = {p.name: p.default for p in kernel.params if p.is_constexpr}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands=_OPERANDS), constexprs=constexprs)
    width = constexprs["N"]
    strides = dict(zip(_STRIDE_NAMES, (matmuls[0].reduction, 1, width, 1, width, 1), strict=True))
    return matmuls, plan(matmuls, strides=strides, dims=dims)


class TileRelaxationTest(unittest.TestCase):
    """Verifies that relaxed tile sizes compile and pad appropriately."""

    def test_a_shallow_reduction_step_is_padded(self) -> None:
        """Asserts a reduction step shallower than two blocks is padded to the 64 rows one unrolled group covers."""
        for kernel, depth in ((gemm_half_k, 32), (gemm_quarter_k, 16)):
            with self.subTest(depth=depth):
                matmuls, planned = _compile_helper(kernel)
                launch = planned.launches[0]
                self.assertEqual(matmuls[0].depth, depth)
                self.assertEqual(launch.loop_blocks, 2)
                padded = next(padding for padding in planned.padding if padding.axis == "k_tile")
                self.assertEqual((padded.extent, padded.padded), (depth, 64))
                _, staging = compile_launch(launch)
                self.assertEqual(staging.rows, 32)

    def test_wide_columns_stand_as_written_where_the_image_fits(self) -> None:
        """Asserts a BLOCK_N of 128 runs at the width the kernel asked for, the columns spilling to VMEM."""
        _, planned = _compile_helper(gemm_excessive_n)
        launch = planned.launches[0]
        self.assertEqual(launch.block_n, 128)
        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)

    def test_large_m_decomposes_and_compiles(self) -> None:
        """Asserts that large row blocks (e.g. BLOCK_M=128) decompose to row_blocks and compile."""
        _, planned = _compile_helper(gemm_excessive_m)
        launch = planned.launches[0]
        self.assertLessEqual(launch.block_m, 64)
        self.assertEqual(launch.row_blocks, 4)
        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)

    def test_rows_run_as_the_widest_placeable_divisor(self) -> None:
        """Asserts the machine rows come from what the register file places rather than from a fixed ceiling."""
        # Reads the planner directly, since tl.arange takes a power of two and no Triton kernel can ask for the
        # 160 rows that tell a derived bound from a fixed one.
        tile = MatMul(rows=160, columns=16, depth=64, accumulator="f32", epilogue=frozenset(),
                      prologue=frozenset(), left_source=None, reduction=64, tile_loops=(), nest=("k",),
                      accesses={})
        self.assertEqual(_machine_tile(index=0, matmul=tile)[1], 80)
        self.assertEqual(_machine_tile(index=0, matmul=replace(tile, rows=128))[1], 64)
        self.assertEqual(_machine_tile(index=0, matmul=replace(tile, rows=48))[1], 48)

    def test_grid_axis_stated_by_nothing_faults(self) -> None:
        """Asserts an axis the grid walks is refused rather than read as one tile of it."""
        with self.assertRaises(ProgramError) as ctx:
            _compile_helper(gemm_grid_m)
        self.assertIn("grid walks m", str(ctx.exception))

    def test_grid_axis_stated_at_the_plan_covers_it(self) -> None:
        """Asserts the same kernel plans every row block once its caller states the axis."""
        _, planned = _compile_helper(gemm_grid_m, dims=({"m": 256},))
        launch = planned.launches[0]
        self.assertEqual(launch.block_m, 32)
        self.assertEqual(launch.row_blocks, 8)
        # Reads the folded column loop as its tile while the grid walks the rows.
        self.assertEqual(launch.column_tiles, 4)
        image, _ = compile_launch(launch)
        self.assertGreater(len(image), 0)

    def test_unplaceable_columns_fault_with_the_measured_image(self) -> None:
        """Asserts an image no placement fits in IRAM is refused, counting the bundles that prove it."""
        _, planned = _compile_helper(gemm_unplaceable_n)
        with self.assertRaises(ProgramError) as ctx:
            compile_launch(planned.launches[0])
        message = str(ctx.exception)
        self.assertIn("Unable to fit the looped_gemm image in IRAM", message)
        self.assertIn("under the hand placement and under packing alike", message)

    def test_retiling_rows_under_a_reducing_epilogue_faults(self) -> None:
        """Asserts rows are not split under an epilogue that reads the whole tile rather than one element."""
        # Reads the planner directly, since plan refuses a reducing epilogue ahead of the guard asserted here.
        reducing = MatMul(rows=128, columns=16, depth=64, accumulator="f32",
                          epilogue=frozenset({"arith.maxnumf", "arith.addf"}), prologue=frozenset(),
                          left_source=None, reduction=64, tile_loops=(), nest=("k",), accesses={})
        with self.assertRaises(ProgramError) as ctx:
            _machine_tile(index=0, matmul=reducing)
        self.assertIn("once per machine tile", str(ctx.exception))

        elementwise = replace(reducing, epilogue=frozenset())
        self.assertEqual(_machine_tile(index=0, matmul=elementwise)[1], 64)

    def test_single_iteration_folded_loop_plans(self) -> None:
        """Asserts that a kernel whose loop runs once (folded by Triton) plans without error."""
        matmuls, planned = _compile_helper(gemm_folded_m)
        launch = planned.launches[0]
        self.assertEqual(matmuls[0].rows, 64)
        self.assertEqual(launch.block_m, 64)
        self.assertEqual(launch.row_blocks, 1)
        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)
        self.assertEqual(staging.rows, 64)


if __name__ == "__main__":
    unittest.main()

