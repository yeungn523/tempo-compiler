"""Contains tests for tile size relaxation and boundary enforcement.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "board_package" / "minitpu_runtime"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from compiler import ProgramError, compile_launch, plan, read_program, signature_of  # noqa: E402


@triton.jit
def gemm_half_k(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 32):
    """Multiplies with a reduction step of 32, which the planner pads to the 64-depth machine quantum."""
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
def gemm_wide_n(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                M: tl.constexpr = 64, N: tl.constexpr = 128, K: tl.constexpr = 192,
                BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 64, BLOCK_K: tl.constexpr = 64):
    """Multiplies with a wide N tile of 64."""
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


_OPERANDS = {"ap": "*bf16", "wp": "*bf16", "op": "*fp32"}
_STRIDE_NAMES = ("stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn")


def _compile_helper(kernel):
    constexprs = {p.name: p.default for p in kernel.params if p.is_constexpr}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands=_OPERANDS), constexprs=constexprs)
    width = constexprs["N"]
    strides = dict(zip(_STRIDE_NAMES, (matmuls[0].reduction, 1, width, 1, width, 1), strict=True))
    return matmuls, plan(matmuls, strides=strides)


class TileRelaxationTest(unittest.TestCase):
    """Verifies that relaxed tile sizes compile and pad appropriately."""

    def test_sub_depth_reduction_padded(self) -> None:
        """Asserts a reduction step of 32 is padded to the 64-depth machine quantum."""
        matmuls, planned = _compile_helper(gemm_half_k)
        launch = planned.launches[0]
        self.assertEqual(matmuls[0].depth, 32)
        self.assertEqual(launch.loop_blocks, 2)  # 2 blocks * 32 = 64 depth
        pad_axes = [p.axis for p in planned.padding]
        self.assertIn("k_tile", pad_axes)
        k_pad = next(p for p in planned.padding if p.axis == "k_tile")
        self.assertEqual(k_pad.extent, 32)
        self.assertEqual(k_pad.padded, 64)

        # Asserts image compilation succeeds
        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)
        self.assertEqual(staging.rows, 32)

    def test_quarter_depth_reduction_padded(self) -> None:
        """Asserts a reduction step of 16 is padded to the 64-depth machine quantum."""
        matmuls, planned = _compile_helper(gemm_quarter_k)
        launch = planned.launches[0]
        self.assertEqual(matmuls[0].depth, 16)
        self.assertEqual(launch.loop_blocks, 2)
        pad_axes = [p.axis for p in planned.padding]
        self.assertIn("k_tile", pad_axes)
        k_pad = next(p for p in planned.padding if p.axis == "k_tile")
        self.assertEqual(k_pad.extent, 16)
        self.assertEqual(k_pad.padded, 64)

        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)
        self.assertEqual(staging.rows, 32)

    def test_wide_tile_compiles(self) -> None:
        """Asserts that BLOCK_N=64 compiles successfully."""
        _, planned = _compile_helper(gemm_wide_n)
        launch = planned.launches[0]
        image, staging = compile_launch(launch)
        self.assertGreater(len(image), 0)
        self.assertEqual(launch.block_n, 64)

    def test_large_tile_decomposes_and_compiles(self) -> None:
        """Asserts that large tile sizes (e.g. BLOCK_N=128) decompose to hardware-safe micro-tiles and compile."""
        _, planned = _compile_helper(gemm_excessive_n)
        launch = planned.launches[0]
        self.assertLessEqual(launch.block_n, 128)
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

