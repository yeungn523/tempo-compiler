"""Contains tests for the pointwise path, from a kernel with no matmul through to the image its plan emits."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as neither the board package nor the code generator
# is installed.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tools"))
sys.path.insert(0, str(_HERE / "board_package"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import build_gpt2_images  # noqa: E402
from codegen import ProgramError, plan_pointwise, read_pointwise, signature_of  # noqa: E402
from codegen.emit import POINTWISE_LEFT, POINTWISE_RIGHT, VectorOp  # noqa: E402
from tests import hand_images  # noqa: E402
import gpt2_triton  # noqa: E402


class PointwiseReaderTest(unittest.TestCase):
    """Covers what the reader recovers from a kernel with no matmul, and what it refuses."""

    def test_a_bias_row_is_read_as_a_broadcast_right_operand(self) -> None:
        """Asserts a row loaded once and expanded down the tokens is the right operand, broadcast."""
        pointwise = read(gpt2_triton.gpt2_add_bias)
        self.assertTrue(pointwise.broadcast_right)
        self.assertEqual(pointwise.features, 768)
        self.assertEqual([operation.name for operation in pointwise.chain], ["arith.addf"])

    def test_two_tensors_are_read_as_full_operands(self) -> None:
        """Asserts two operands addressed as the store is are both read in full."""
        pointwise = read(gpt2_triton.gpt2_add)
        self.assertFalse(pointwise.broadcast_right)
        self.assertEqual(pointwise.chain[0].operands, pointwise.operands)

    def test_the_width_follows_the_kernel(self) -> None:
        """Asserts the features come from the store's row stride, which the kernel's N sets."""
        self.assertEqual(read(gpt2_triton.gpt2_add_bias, features=3072).features, 3072)

    def test_a_kernel_the_image_cannot_run_is_refused(self) -> None:
        """Asserts each kernel breaking one rule of the pointwise image raises rather than compiling."""
        for kernel in (pointwise_reduce, pointwise_masked, pointwise_constant, pointwise_transposed,
                       pointwise_partial_row, pointwise_one_operand):
            with self.subTest(kernel=kernel.fn.__name__), self.assertRaises(ProgramError):
                read(kernel)


class PointwiseLoweringTest(unittest.TestCase):
    """Covers the vector operations a pointwise body lowers to."""

    def test_an_add_lowers_to_one_vadd_of_the_operand_registers(self) -> None:
        """Asserts the residual add is one vadd reading the left and right registers."""
        launch = plan_pointwise(read(gpt2_triton.gpt2_add))
        self.assertEqual(launch.body, (VectorOp(kind="vadd", into=2, sources=(POINTWISE_LEFT, POINTWISE_RIGHT)),))

    def test_a_division_lowers_to_a_reciprocal_and_a_multiply(self) -> None:
        """Asserts a division runs as the divisor's reciprocal times the dividend, the machine carrying no divider."""
        launch = plan_pointwise(read(pointwise_divide))
        self.assertEqual([operation.kind for operation in launch.body], ["vrecip", "vmul"])
        self.assertEqual(launch.body[0].sources, (POINTWISE_RIGHT,))

    def test_a_chain_lowers_in_program_order(self) -> None:
        """Asserts an exponential of a difference runs the subtraction first, each value in a register of its own."""
        body = plan_pointwise(read(pointwise_exp_difference)).body
        self.assertEqual([operation.kind for operation in body], ["vsub", "vexp"])
        self.assertEqual(body[1].sources, (body[0].into,))


class PointwiseImageTest(unittest.TestCase):
    """Holds every pointwise image a GPT-2 layer launches to the hand-written builder it replaces."""

    def test_every_gpt2_pointwise_image_matches_the_hand_written_builder(self) -> None:
        """Asserts each compiled add assembles to the bytes the hand-written builder did for the same width."""
        for kind, features in build_gpt2_images._POINTWISE:
            with self.subTest(kind=kind, features=features):
                image, entry = build_gpt2_images._compile_pointwise(kind=kind, features=features)
                self.assertEqual(image, hand_images.elementwise_add(entry["chunks"],
                                                                    broadcast_right=entry["broadcast_right"]))


def read(kernel, *, features: int = 768):
    """Returns what the reader recovers from a pointwise kernel at the given width."""
    constexprs = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    return read_pointwise(kernel, signature=signature_of(kernel, operands={}), constexprs=constexprs | {"N": features})


@triton.jit
def pointwise_reduce(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                     N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Sums each row, which combines elements across positions."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        total = tl.sum(tl.load(left_pointer + offsets) + tl.load(right_pointer + offsets), 1)
        tl.store(output_pointer + offs_m * N, total)


@triton.jit
def pointwise_masked(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                     N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Adds under a mask, which the host's whole tiles leave the image no way to honor."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        offsets = offs_m[:, None] * N + offs_n[None, :]
        inside = offs_n[None, :] < N
        total = tl.load(left_pointer + offsets, mask=inside, other=0.0) + tl.load(right_pointer + offsets)
        tl.store(output_pointer + offsets, total)


@triton.jit
def pointwise_constant(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                       N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Doubles one operand, a number the image holds nowhere."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) * 2.0 + tl.load(right_pointer + offsets))


@triton.jit
def pointwise_transposed(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                         N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 32):
    """Adds the right operand read at its transposed position, which no element of the left shares."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        offsets = offs_m[:, None] * N + offs_n[None, :]
        transposed = offs_m[:, None] + offs_n[None, :] * N
        tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) + tl.load(right_pointer + transposed))


@triton.jit
def pointwise_partial_row(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                          N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Adds the first half of each row only, which leaves the rest of the row unwritten."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N // 2, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) + tl.load(right_pointer + offsets))


@triton.jit
def pointwise_one_operand(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                          N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Exponentiates one operand, where the image stages two."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, tl.exp(tl.load(left_pointer + offsets).to(tl.float32)).to(tl.bfloat16))


@triton.jit
def pointwise_divide(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                     N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Divides one operand by the other."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, tl.load(left_pointer + offsets) / tl.load(right_pointer + offsets))


@triton.jit
def pointwise_exp_difference(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                             N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Exponentiates the difference of the two operands."""
    offs_m = tl.arange(0, 32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        difference = (tl.load(left_pointer + offsets) - tl.load(right_pointer + offsets)).to(tl.float32)
        tl.store(output_pointer + offsets, tl.exp(difference).to(tl.bfloat16))


if __name__ == "__main__":
    unittest.main()
