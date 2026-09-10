"""Contains tests for the compiler, from a decorated Triton kernel through to the image its plan emits.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as neither the board package nor the compiler is
# installed.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
"""The repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tools"))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "board_package" / "minitpu_runtime"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import kernels  # noqa: E402
from compiler import ProgramError, plan, read_program, signature_of  # noqa: E402
from compiler_tests.test_layout import renaming_onto  # noqa: E402
from layout import register_map  # noqa: E402

_OPERANDS: dict[str, str] = {"ap": "*bf16", "wp": "*bf16", "op": "*fp32"}
"""Triton type of each parameter a buffer is passed for, which every kernel below shares."""

_STRIDE_NAMES: tuple[str, ...] = ("stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn")
"""The stride parameters every kernel below declares, in the order it declares them."""


class ProgramReaderTest(unittest.TestCase):
    """Covers what `read_program` recovers from a kernel's lowered module."""

    def test_the_matmul_carries_the_block_the_kernel_states(self) -> None:
        """Asserts the recovered matmul carries the tile extents and the contraction the kernel declares."""
        matmul = read_program(
            gemm, signature=signature_of(gemm, operands=_OPERANDS),
            constexprs={parameter.name: parameter.default for parameter in gemm.params
                        if parameter.is_constexpr})[0]
        self.assertEqual((matmul.rows, matmul.columns, matmul.depth), (32, 16, 64))
        self.assertEqual(matmul.reduction, 192)
        self.assertEqual(matmul.accumulator, "f32")
        self.assertEqual(tuple(loop.step for loop in matmul.tile_loops), (32, 16))

    def test_a_token_loop_stepping_like_the_reduction_is_refused(self) -> None:
        """Asserts a nest whose loops cannot be told apart by their step is refused rather than misread.

        The reader names each axis by the step of the loop walking it, so a token block the size of the reduction
        block leaves two loops indistinguishable. Recovering the axes from the addresses each loop advances is what
        would lift this.
        """
        with self.assertRaises(ProgramError):
            compile_one(gemm_tall_rows)


class PlannerTest(unittest.TestCase):
    """Covers the launches `plan` derives from a matmul."""

    def test_the_launch_divides_each_extent_by_its_machine_unit(self) -> None:
        """Asserts each launch count is the axis extent over the unit the machine covers it in."""
        matmuls, planned = compile_one(gemm)
        launch = planned.launches[0]
        self.assertEqual(launch.row_blocks, 2)
        self.assertEqual(launch.column_tiles, 3)
        self.assertEqual(launch.groups, 3)
        self.assertEqual(launch.loop_blocks, 2)
        self.assertFalse(launch.activate)
        self.assertFalse(launch.fuse_groups)
        self.assertEqual(planned.issued_over_useful, 1.0)
        self.assertEqual(planned.padding, ())
        self.assertEqual(planned.repack, ())

    def test_the_drain_matches_the_partials_the_launch_leaves(self) -> None:
        """Asserts the drain covers one partial per group per column tile when the groups are not fused."""
        _, planned = compile_one(gemm)
        launch, drain = planned.launches[0], planned.drains[0]
        self.assertEqual(drain.shape[:2], (launch.column_tiles, launch.groups))

    def test_a_block_wider_than_the_array_is_planned_as_its_column_tiles(self) -> None:
        """Asserts a block covering several column tiles is planned, since the register file holds its accumulators.

        The array is one tile wide, so a wider block is that many tiles rather than something it cannot compute.
        What bounds the width is `layout.register_map`, which `plan` asks rather than naming a width outright.
        """
        matmuls, planned = compile_one(gemm_wide_columns, dims=({"m": 32},))
        self.assertEqual(matmuls[0].columns, 32)
        self.assertEqual(planned.launches[0].column_tiles, 4)
        self.assertEqual(register_map(block_m=matmuls[0].rows, block_n=matmuls[0].columns).resident, 2)


class PipelineTest(unittest.TestCase):
    """Covers the whole path, holding the image a plan emits to the one the hand-written builder assembles."""

    def test_the_planned_image_matches_the_hand_written_builder(self) -> None:
        """Asserts a compiled kernel emits the bundle stream `build_looped_gemm` assembles for the same launch."""
        from sim_triton_kernel import compile_launch  # noqa: PLC0415  -- a tool, not a package module

        _, planned = compile_one(gemm)
        launch = planned.launches[0]
        hand = kernels.build_looped_gemm(launch.column_tiles, launch.groups, activate=launch.activate,
                                         fuse_groups=launch.fuse_groups,
                                         loop_blocks=launch.loop_blocks).image.assemble(pad_to=4)
        emitted, _ = compile_launch(launch)
        derived = tuple(int.from_bytes(emitted[at:at + 16], "little") for at in range(0, len(emitted), 16))
        self.assertIsNotNone(renaming_onto(derived, hand))

    def test_every_field_the_image_carries_reaches_it(self) -> None:
        """Asserts each launch field describing the image changes it, so none is dropped before the emitter.

        A field the emitter never reads leaves the image the plan's own value would have produced, which every
        other check passes over because the image it compares is still self consistent.
        """
        import dataclasses  # noqa: PLC0415  -- used only here
        from sim_triton_kernel import compile_launch  # noqa: PLC0415  -- a tool, not a package module

        # Covers a whole number of two tile blocks, so widening the block stays legal.
        base = dataclasses.replace(compile_one(gemm)[1].launches[0], column_tiles=4)
        reference, _ = compile_launch(base)
        for field, value in (("column_tiles", 8), ("groups", 6), ("loop_blocks", 4),
                             ("fuse_groups", True), ("block_m", 16), ("block_n", 32)):
            with self.subTest(field=field):
                altered, _ = compile_launch(dataclasses.replace(base, **{field: value}))
                self.assertNotEqual(altered, reference)

    def test_the_launch_count_stays_off_the_image(self) -> None:
        """Asserts `row_blocks` leaves the image alone, since it counts the launches a host issues."""
        import dataclasses  # noqa: PLC0415  -- used only here
        from sim_triton_kernel import compile_launch  # noqa: PLC0415  -- a tool, not a package module

        base = dataclasses.replace(compile_one(gemm)[1].launches[0], column_tiles=4)
        self.assertEqual(compile_launch(dataclasses.replace(base, row_blocks=4))[0],
                         compile_launch(base)[0])

    def test_a_launch_that_activates_is_refused(self) -> None:
        """Asserts a launch carrying an epilogue raises, since the emitter places the plain product only."""
        import dataclasses  # noqa: PLC0415  -- used only here
        from sim_triton_kernel import compile_launch  # noqa: PLC0415  -- a tool, not a package module

        base = compile_one(gemm)[1].launches[0]
        with self.assertRaises(NotImplementedError):
            compile_launch(dataclasses.replace(base, activate=True))

    def test_a_wide_block_reaches_the_emitter(self) -> None:
        """Asserts the block a kernel names changes the image, rather than being dropped on the way to the emitter.

        A block covering several tiles emits a body holding all of them, so an image matching the one tile wide
        block's is the width going missing somewhere between the plan and the emitter.
        """
        import asm  # noqa: PLC0415  -- imported here so the path insert above has already run
        import emit  # noqa: PLC0415
        from sim_triton_kernel import compile_launch  # noqa: PLC0415  -- a tool, not a package module

        launch = compile_one(gemm_wide_columns, dims=({"m": 32},))[1].launches[0]
        self.assertEqual(launch.block_n, 32)
        # Serialised the same way the tool does, so padding cannot make the two differ on its own.
        narrow = asm.AsmKernel(name="looped_gemm", bundles=emit.looped_gemm_stream(
            column_tiles=launch.column_tiles, groups=launch.groups, loop_blocks=launch.loop_blocks,
            fuse_groups=launch.fuse_groups)[0]).to_bytes(pad_to=4)
        self.assertNotEqual(compile_launch(launch)[0], narrow)

    def test_a_wide_block_keeps_an_accumulator_set_per_tile(self) -> None:
        """Asserts a wide block's image names the registers the map lays out for it, which is one set per tile."""
        import asm  # noqa: PLC0415  -- imported here so the path insert above has already run
        import bf16  # noqa: PLC0415
        import emit  # noqa: PLC0415

        launch = compile_one(gemm_wide_columns, dims=({"m": 32},))[1].launches[0]
        words, _ = emit.looped_gemm_stream(column_tiles=launch.column_tiles, groups=launch.groups,
                                           loop_blocks=launch.loop_blocks, fuse_groups=launch.fuse_groups,
                                           block_n=launch.block_n)
        touched: set[int] = set()
        for word in words:
            decoded = asm._decode(word)
            touched |= set(decoded["reads"]) | {register for register, _ in decoded["writes"]}
        mapped = register_map(block_m=launch.block_m, block_n=launch.block_n)
        per_tile = mapped.token_groups * bf16._TILE_VREGS
        self.assertEqual(mapped.resident, 2)
        self.assertEqual(touched, set(range(mapped.accumulator, mapped.result + bf16._TILE_VREGS)))
        self.assertEqual(len(touched), mapped.resident * per_tile + per_tile + 2 * bf16._TILE_VREGS)


@triton.jit
def gemm(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
         M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
         BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies an M by K activation by a K by N weight, one block of the output at a time."""
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
def gemm_wide_columns(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      M: tl.constexpr = 32, N: tl.constexpr = 64, K: tl.constexpr = 192,
                      BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 32, BLOCK_K: tl.constexpr = 64):
    """Multiplies with a block twice as wide as the systolic array, over one token block."""
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
def gemm_tall_rows(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                   BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies with a token block that steps by the same amount the reduction does."""
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


def compile_one(kernel, *, dims: tuple[dict[str, int], ...] = ()):
    """Returns the matmuls one kernel states and the launches a plan makes of them.

    Args:
        kernel: The function the triton.jit decorator returned.
        dims: Extent each matmul covers along an axis its module does not state, which a loop running once leaves
            behind when canonicalization folds it.

    Returns:
        The matmuls and the plan.
    """
    constexprs = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands=_OPERANDS), constexprs=constexprs)
    width = constexprs["N"]
    strides = dict(zip(_STRIDE_NAMES, (matmuls[0].reduction, 1, width, 1, width, 1), strict=True))
    return matmuls, plan(matmuls, strides=strides, dims=dims)


if __name__ == "__main__":
    unittest.main()
