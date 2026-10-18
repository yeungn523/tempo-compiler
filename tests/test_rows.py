"""Contains tests for the row path, from a kernel summing over each token's features through to the image it emits."""

from __future__ import annotations

import collections
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

import numpy as np  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import build_gpt2_images  # noqa: E402
from codegen import ProgramError, plan_rows, read_rows, signature_of  # noqa: E402
from codegen.asm import AsmKernel  # noqa: E402
from codegen.emit import RowLoad, RowStore, VectorOp, row_stream  # noqa: E402
from tests import hand_images  # noqa: E402
from tests.test_emit import _e_opcode  # noqa: E402
import gpt2_triton  # noqa: E402


class RowReaderTest(unittest.TestCase):
    """Covers what the reader recovers from a kernel summing over each token's features, and what it refuses."""

    def test_layernorm_is_read_as_two_nested_sums(self) -> None:
        """Asserts the variance's sum reads the mean's, and the output reads gamma and beta as per-feature rows."""
        rows = read(gpt2_triton.gpt2_layernorm)
        self.assertEqual(rows.rows, ("gamma_pointer", "beta_pointer"))
        self.assertEqual(rows.features, 768)
        sums = [term for term in walk(rows.output) if term.name == "sum"]
        self.assertEqual(len(set(sums)), 2)
        self.assertTrue(any(inner in walk(outer.operands[0]) for outer in sums for inner in sums if inner != outer))

    def test_a_whole_row_tile_is_summed_without_a_loop(self) -> None:
        """Asserts a tile as wide as the row is summed directly, with no running total to fold it into."""
        sums = [term for term in walk(read(rows_share, features=256).output) if term.name == "sum"]
        self.assertEqual(sums[0].operands[0].name, "input")

    def test_a_kernel_the_image_cannot_run_is_refused(self) -> None:
        """Asserts each kernel breaking one rule of the row image raises rather than compiling."""
        for kernel, features in ((rows_maximum, 768), (rows_token_sum, 256), (rows_partial_sum, 768),
                                 (rows_two_inputs, 256), (rows_square_root, 256)):
            with self.subTest(kernel=kernel.fn.__name__), self.assertRaises(ProgramError):
                read(kernel, features=features)


class RowLoweringTest(unittest.TestCase):
    """Covers the operations a row kernel lowers to and the registers they are given."""

    def test_layernorm_stages_the_reciprocal_width_and_the_epsilon(self) -> None:
        """Asserts the divisions by the width become products with a staged reciprocal, and no vrecip runs."""
        launch = plan_rows(read(gpt2_triton.gpt2_layernorm))
        self.assertEqual(launch.rows, 2)
        np.testing.assert_allclose(launch.scalars, (1.0 / 768, 1e-5), rtol=1e-6)
        self.assertNotIn("vrecip", [operation.kind for operation in launch.body if isinstance(operation, VectorOp)])

    def test_each_sum_reduces_every_vreg_and_the_output_stores_every_vreg(self) -> None:
        """Asserts the two sums run one vredsum per VREG of the row, and the output pass stores each VREG once."""
        body = plan_rows(read(gpt2_triton.gpt2_layernorm)).body
        kinds = collections.Counter(operation.kind for operation in body if isinstance(operation, VectorOp))
        self.assertEqual(kinds["vredsum"], 24)
        self.assertEqual([operation.tile for operation in body if isinstance(operation, RowStore)], list(range(12)))

    def test_a_division_by_a_sum_lowers_to_a_reciprocal_and_a_multiply(self) -> None:
        """Asserts a divisor one token shares runs as its reciprocal, taken once, times every element."""
        kinds = [operation.kind for operation in plan_rows(read(rows_share, features=256)).body
                 if isinstance(operation, VectorOp)]
        self.assertEqual(kinds.count("vrecip"), 1)
        self.assertEqual(kinds.count("vmul"), 4)

    def test_every_register_lies_in_the_register_file(self) -> None:
        """Asserts the assignment names only VREGs the machine holds, the constants keeping the lowest ones."""
        launch = plan_rows(read(gpt2_triton.gpt2_layernorm))
        self.assertEqual([load.into for load in launch.prologue], [0, 1])
        named = {load.into for load in launch.prologue}
        for operation in launch.body:
            if isinstance(operation, RowStore):
                named.add(operation.source)
            else:
                named.add(operation.into)
            named.update(getattr(operation, "sources", ()))
        self.assertLessEqual(max(named), 31)
        self.assertFalse({operation.into for operation in launch.body if isinstance(operation, (RowLoad, VectorOp))}
                         & {0, 1})


class RowImageTest(unittest.TestCase):
    """Holds the compiled LayerNorm to the hand-written builder it replaces."""

    def test_layernorm_runs_the_hand_written_arithmetic(self) -> None:
        """Asserts the image runs the hand builder's E-slot operations, less the two adds seeding its sums from zero."""
        launch = plan_rows(read(gpt2_triton.gpt2_layernorm))
        compiled = row_stream(features=launch.features, rows=launch.rows, scalars=len(launch.scalars),
                              prologue=launch.prologue, body=launch.body)
        image = hand_images.layernorm_rows()
        hand = [int.from_bytes(image[at:at + 16], "little") for at in range(0, len(image), 16)]
        compiled_opcodes = collections.Counter(_e_opcode(word) for word in compiled if _e_opcode(word))
        hand_opcodes = collections.Counter(_e_opcode(word) for word in hand if _e_opcode(word))
        vadd = _e_opcode(next(word for word in row_stream(
            features=64, rows=0, scalars=0, prologue=(),
            body=(VectorOp(kind="vadd", into=0, sources=(0, 0)),)) if _e_opcode(word)))
        self.assertEqual(compiled_opcodes + collections.Counter({vadd: 2}), hand_opcodes)

    def test_the_gpt2_layernorm_image_fits_and_names_its_constants(self) -> None:
        """Asserts the build tool's image assembles under packing and its entry lists the rows and numbers."""
        image, entry = build_gpt2_images._compile_rows(kind="layernorm", features=768)
        self.assertEqual(entry["rows"], ["gamma_pointer", "beta_pointer"])
        self.assertEqual(len(entry["scalars"]), 2)
        launch = plan_rows(read(gpt2_triton.gpt2_layernorm))
        bundles = row_stream(features=768, rows=2, scalars=2, prologue=launch.prologue, body=launch.body)
        self.assertEqual(image, AsmKernel(name="rows", bundles=bundles, placement="pack").to_bytes(pad_to=4))


def read(kernel, *, features: int = 768):
    """Returns what the reader recovers from a row kernel at the given width."""
    constexprs = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    return read_rows(kernel, signature=signature_of(kernel, operands={}), constexprs=constexprs | {"N": features})


def walk(term):
    """Returns every term one term is built from, itself included."""
    found, pending = [], [term]
    while pending:
        current = pending.pop()
        found.append(current)
        pending.extend(current.operands)
    return found


@triton.jit
def rows_share(input_pointer: "*bf16", output_pointer: "*bf16", N: tl.constexpr = 256):
    """Divides each element by its token's total, the tile covering the whole row."""
    offs_m = tl.arange(0, 32)
    offsets = offs_m[:, None] * N + tl.arange(0, N)[None, :]
    values = tl.load(input_pointer + offsets).to(tl.float32)
    tl.store(output_pointer + offsets, (values / tl.sum(values, 1)[:, None]).to(tl.bfloat16))


@triton.jit
def rows_maximum(input_pointer: "*bf16", output_pointer: "*bf16", N: tl.constexpr = 768, BLOCK_N: tl.constexpr = 256):
    """Subtracts each token's maximum, a reduce the image folds by nothing but addition."""
    offs_m = tl.arange(0, 32)
    largest = tl.full((32, BLOCK_N), -1e30, dtype=tl.float32)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        largest = tl.maximum(largest, tl.load(input_pointer + offsets).to(tl.float32))
    peak = tl.max(largest, 1)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, (tl.load(input_pointer + offsets) - peak[:, None]).to(tl.bfloat16))


@triton.jit
def rows_token_sum(input_pointer: "*bf16", output_pointer: "*bf16", N: tl.constexpr = 256):
    """Subtracts each feature's total over the tokens, which sums along the wrong axis."""
    offs_m = tl.arange(0, 32)
    offsets = offs_m[:, None] * N + tl.arange(0, N)[None, :]
    values = tl.load(input_pointer + offsets).to(tl.float32)
    tl.store(output_pointer + offsets, (values - tl.sum(values, 0)[None, :]).to(tl.bfloat16))


@triton.jit
def rows_partial_sum(input_pointer: "*bf16", output_pointer: "*bf16", N: tl.constexpr = 768,
                     BLOCK_N: tl.constexpr = 256):
    """Subtracts the total of the first half of each row, a sum whose loop stops short of the row's end."""
    offs_m = tl.arange(0, 32)
    total = tl.zeros((32, BLOCK_N), dtype=tl.float32)
    for tile in range(0, N // 2, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        total += tl.load(input_pointer + offsets).to(tl.float32)
    half = tl.sum(total, 1)
    for tile in range(0, N, BLOCK_N):
        offsets = offs_m[:, None] * N + (tile + tl.arange(0, BLOCK_N))[None, :]
        tl.store(output_pointer + offsets, (tl.load(input_pointer + offsets) - half[:, None]).to(tl.bfloat16))


@triton.jit
def rows_two_inputs(left_pointer: "*bf16", right_pointer: "*bf16", output_pointer: "*bf16",
                    N: tl.constexpr = 256):
    """Subtracts one tensor's row total from another tensor, where the image streams one."""
    offs_m = tl.arange(0, 32)
    offsets = offs_m[:, None] * N + tl.arange(0, N)[None, :]
    total = tl.sum(tl.load(left_pointer + offsets).to(tl.float32), 1)
    tl.store(output_pointer + offsets, (tl.load(right_pointer + offsets) - total[:, None]).to(tl.bfloat16))


@triton.jit
def rows_square_root(input_pointer: "*bf16", output_pointer: "*bf16", N: tl.constexpr = 256):
    """Divides by the square root of each token's total, an operation the vector unit runs only reciprocated."""
    offs_m = tl.arange(0, 32)
    offsets = offs_m[:, None] * N + tl.arange(0, N)[None, :]
    values = tl.load(input_pointer + offsets).to(tl.float32)
    tl.store(output_pointer + offsets, (values / tl.sqrt(tl.sum(values, 1))[:, None]).to(tl.bfloat16))


if __name__ == "__main__":
    unittest.main()
