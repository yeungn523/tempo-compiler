"""Contains tests for the compiler, from a decorated Triton kernel through to the image its plan emits.
"""

from __future__ import annotations

import dataclasses
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
import sample_triton  # noqa: E402
from bf16 import _BLOCK_DEPTH, _BLOCK_TOKENS, _TILE  # noqa: E402
from compiler import ProgramError, compile_launch, plan, read_program, signature_of  # noqa: E402
from compiler.allocation import COMMIT, allocate  # noqa: E402
import emit  # noqa: E402
from compiler.dataflow import (CARRIED, CONSTANT, STREAMED, TRANSIENT, Trip,  # noqa: E402
                               Value, attention_trip)
from compiler.ir import FlashLaunch, FusedLaunch, Operation  # noqa: E402
from compiler_tests.test_layout import renaming_onto  # noqa: E402
from layout import flash_register_map, register_map  # noqa: E402

_OPERANDS: dict[str, str] = {"ap": "*bf16", "wp": "*bf16", "op": "*fp32"}

_ATTENTION_OPERANDS: dict[str, str] = {"qp": "*bf16", "kp": "*bf16", "vp": "*bf16", "op": "*fp32"}
"""Triton type of each buffer an attention kernel is compiled against."""
"""Triton type of each parameter a buffer is passed for, which every kernel below shares."""

_STRIDE_NAMES: tuple[str, ...] = ("stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn")
"""The stride parameters every kernel below declares, in the order it declares them."""

_ROOT_TWO: tl.constexpr = tl.constexpr(1.4142135623730951)
"""Divisor the exact GELU scales its argument by, which Triton reaches from a kernel only as a constexpr."""


class SignatureTest(unittest.TestCase):
    """Covers the signature a kernel is compiled against, and what states it."""

    def test_a_kernel_states_its_own_operands(self) -> None:
        """Asserts an annotated pointer is typed from the annotation, with nothing passed for it."""
        for kernel, pointers in ((sample_triton.matmul_kernel, {
                "activation_pointer": "*bf16", "weight_pointer": "*bf16", "output_pointer": "*fp32"}),
                (sample_triton.fused_ffn_kernel, {
                    "activation_pointer": "*bf16", "up_weight_pointer": "*bf16",
                    "down_weight_pointer": "*bf16", "output_pointer": "*fp32"})):
            with self.subTest(kernel=kernel.__name__):
                signature = signature_of(kernel, operands={})
                self.assertEqual({name: kind for name, kind in signature.items() if kind != "constexpr"},
                                 pointers)

    def test_an_argument_overrides_the_annotation(self) -> None:
        """Asserts a named operand wins over what the kernel annotated, which is what the flag is for."""
        signature = signature_of(sample_triton.matmul_kernel, operands={"weight_pointer": "*fp32"})
        self.assertEqual(signature["weight_pointer"], "*fp32")
        self.assertEqual(signature["activation_pointer"], "*bf16")

    def test_an_unannotated_parameter_stays_a_scalar(self) -> None:
        """Asserts a runtime parameter no annotation and no argument names is typed as an extent."""
        self.assertEqual(signature_of(gemm, operands={})["ap"], "i32")


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

    def test_a_shared_token_step_is_read(self) -> None:
        """Asserts a token block the size of the reduction block leaves the reduction named by its addresses."""
        matmuls, planned = compile_one(gemm_tall_rows, dims=({"m": 64},))
        matmul = matmuls[0]
        self.assertEqual((matmul.rows, matmul.columns, matmul.depth), (64, 16, 64))
        self.assertEqual(matmul.reduction, 192)
        self.assertEqual(tuple(loop.axis for loop in matmul.tile_loops), ("n",))
        self.assertEqual(planned.launches[0].row_blocks, 1)

    def test_a_shared_column_step_is_read(self) -> None:
        """Asserts a block as wide as its reduction is deep leaves every loop's axis recovered."""
        matmuls, planned = compile_one(gemm_square_tile)
        matmul = matmuls[0]
        self.assertEqual((matmul.rows, matmul.columns, matmul.depth), (32, 64, 64))
        self.assertEqual(matmul.reduction, 192)
        self.assertEqual(tuple(loop.axis for loop in matmul.tile_loops), ("m", "n"))
        launch = planned.launches[0]
        self.assertEqual((launch.row_blocks, launch.column_tiles, launch.groups), (2, 8, 3))

    def test_a_folded_reduction_is_read_from_its_operands(self) -> None:
        """Asserts a kernel whose reduction ran once contracts the extent its operands meet at."""
        matmuls, planned = compile_one(gemm_folded_reduction)
        matmul = matmuls[0]
        self.assertEqual((matmul.rows, matmul.columns, matmul.depth), (32, 16, 64))
        self.assertEqual(matmul.reduction, 64)
        self.assertEqual(tuple(loop.axis for loop in matmul.tile_loops), ("m", "n"))
        launch = planned.launches[0]
        self.assertEqual((launch.row_blocks, launch.column_tiles, launch.groups), (2, 3, 1))

    def test_an_attention_block_is_read_as_its_two_products(self) -> None:
        """Asserts a flash attention kernel reads as the score product and the product consuming it."""
        matmuls = read_attention()
        scores, attended = matmuls
        self.assertEqual((scores.rows, scores.columns, scores.depth), (16, 16, 64))
        self.assertIsNone(scores.left_source)
        self.assertEqual(scores.nest, ("m", "n", "k"))
        self.assertEqual((attended.rows, attended.columns, attended.depth), (16, 64, 16))
        self.assertEqual(attended.left_source, 0)
        self.assertEqual(attended.nest, ("k",))

    def test_a_reduce_is_read_as_the_operation_it_folds_under(self) -> None:
        """Asserts a row maximum and a row sum are told apart by the combiner each reduce carries."""
        matmuls = read_attention()
        self.assertIn("tt.reduce.arith.maxnumf", matmuls[1].prologue)
        self.assertIn("tt.reduce.arith.addf", matmuls[1].epilogue)
        self.assertNotIn("tt.reduce.arith.addf", matmuls[1].prologue)

    def test_the_key_loop_carries_the_statistics_and_the_accumulator(self) -> None:
        """Asserts the values the key loop hands on are read, the accumulator told apart by its rank."""
        scores, attended = read_attention()
        self.assertEqual(scores.carried, ())
        self.assertEqual([value.shape for value in attended.carried], [(16,), (16,), (16, 64)])

    def test_the_transform_ahead_of_the_matmul_is_recovered(self) -> None:
        """Asserts the operations standing between the left operand's load and the matmul are read as its prologue."""
        matmul = compile_one(gemm_activated)[0][0]
        self.assertEqual(matmul.prologue, frozenset({"arith.addf", "arith.divf", "arith.mulf", "math.erf"}))
        self.assertEqual(matmul.epilogue, frozenset())
        self.assertIsNone(matmul.left_source)

    def test_a_transform_leaves_the_operand_addressable(self) -> None:
        """Asserts a transform standing ahead of the matmul is walked through to the load the addressing lies in.

        A reader stopping at the transform recovers no activation pointer, which leaves every stride, mask and
        loop axis the plan plans from reading as though the operand came from nowhere.
        """
        plain, activated = compile_one(gemm)[0][0], compile_one(gemm_activated)[0][0]
        self.assertEqual(activated.accesses["activation"], plain.accesses["activation"])
        self.assertEqual(activated.nest, plain.nest)
        self.assertEqual(tuple(loop.axis for loop in activated.tile_loops),
                         tuple(loop.axis for loop in plain.tile_loops))

    def test_a_width_cast_alone_is_no_transform(self) -> None:
        """Asserts the widening around an activation is left out of the prologue the arithmetic states.

        The machine stages one width whatever width the kernel wrote its arithmetic in, so recording the casts
        would key one fused transform under as many sets as there are ways to spell its precision.
        """
        matmul = compile_one(gemm_activated)[0][0]
        self.assertNotIn("arith.extf", matmul.prologue)
        self.assertNotIn("arith.truncf", matmul.prologue)


class AttentionTripTest(unittest.TestCase):
    """Covers the dataflow one trip of an attention block's key loop runs."""

    def test_the_chains_merge_into_one_trip(self) -> None:
        """Asserts the chains standing around the two products merge into one order, each operation once."""
        trip = attention_trip(read_attention(), produces=0, consumes=1)
        positions = [operation.position for operation in trip.operations]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(positions), len(set(positions)))
        self.assertEqual([operation.name for operation in trip.operations],
                         ["tt.dot", "tt.reduce.arith.maxnumf", "arith.maxnumf", "arith.subf", "math.exp",
                          "arith.subf", "math.exp", "arith.mulf", "tt.reduce.arith.addf", "arith.addf",
                          "arith.mulf", "tt.dot"])

    def test_the_trip_names_a_carried_value_by_what_it_does_with_it(self) -> None:
        """Asserts the accumulator, the running maximum and the running sum are told apart by use."""
        scores, attended = read_attention()
        trip = attention_trip((scores, attended), produces=0, consumes=1)
        self.assertEqual([trip.values[carried.leaving].region for carried in attended.carried],
                         ["row_max", "row_sum", "accumulator"])
        for carried in attended.carried:
            region, number = trip.values[carried.leaving].region, trip.values[carried.entering].number
            for identifier in (carried.entering, carried.inside, carried.leaving):
                self.assertEqual(trip.values[identifier], Value(identifier, CARRIED, region, number))

    def test_the_second_product_reads_the_probabilities_through_their_cast(self) -> None:
        """Asserts a width cast between the exponential and the product leaves one value rather than two."""
        trip = attention_trip(read_attention(), produces=0, consumes=1)
        exponentials = [one for one in trip.operations if one.name == "math.exp"]
        attended = trip.operations[-1]
        self.assertEqual(attended.name, "tt.dot")
        self.assertIn(exponentials[0].result, attended.operands)

    def test_a_value_the_trip_computes_is_told_from_one_it_reads(self) -> None:
        """Asserts a trip's own values are transient, its tiles streamed and the numbers it states constant."""
        trip = attention_trip(read_attention(), produces=0, consumes=1)
        produced = {operation.result for operation in trip.operations}
        for identifier, value in trip.values.items():
            if value.role == TRANSIENT:
                self.assertIn(identifier, produced)
            if value.role in (STREAMED, CONSTANT):
                self.assertNotIn(identifier, produced)
        counted = {role: sum(value.role == role for value in trip.values.values())
                   for role in (STREAMED, CONSTANT)}
        self.assertEqual(counted, {STREAMED: 3, CONSTANT: 1})
        self.assertEqual(trip.constants, ())

    def test_the_normalization_stands_outside_the_trip(self) -> None:
        """Asserts the division the kernel ends on runs once the loop ends rather than on every trip."""
        trip = attention_trip(read_attention(), produces=0, consumes=1)
        self.assertEqual([operation.name for operation in trip.after], ["arith.divf"])
        self.assertNotIn("arith.divf", [operation.name for operation in trip.operations])

    def test_a_scale_on_the_scores_reaches_the_trip(self) -> None:
        """Asserts scaling the scores leaves an operation in the trip rather than passing unread."""
        plain = attention_trip(read_attention(), produces=0, consumes=1)
        scaled = attention_trip(read_attention(flash_attention_scaled), produces=0, consumes=1)
        self.assertEqual(len(scaled.operations), len(plain.operations) + 1)
        product, scale = scaled.operations[0], scaled.operations[1]
        self.assertEqual((product.name, scale.name), ("tt.dot", "arith.mulf"))
        self.assertIn(product.result, scale.operands)

    def test_a_matmul_carrying_no_product_is_refused(self) -> None:
        """Asserts a trip is refused where a matmul carries no record of its own dot."""
        scores, attended = read_attention()
        with self.assertRaises(ProgramError) as raised:
            attention_trip((dataclasses.replace(scores, product=None), attended), produces=0, consumes=1)
        self.assertIn("carries no record of its own product", str(raised.exception))

    def test_an_unnamed_carried_value_is_refused(self) -> None:
        """Asserts a trip is refused where the loop carries a value the register file holds nowhere."""
        scores, attended = read_attention()
        spare = dataclasses.replace(attended.carried[1], position=9, entering=-1, inside=-2, leaving=-3)
        with self.assertRaises(ProgramError) as raised:
            attention_trip((scores, dataclasses.replace(attended, carried=(*attended.carried, spare))),
                           produces=0, consumes=1)
        self.assertIn("Unable to name every value the key loop carries", str(raised.exception))


class AttentionAllocationTest(unittest.TestCase):
    """Covers the regions and the order one attention trip's operations are placed under."""

    def test_every_operation_reads_the_region_holding_its_own_value(self) -> None:
        """Asserts replaying the placement finds each operand where the operation reading it looks."""
        for head in (16, 32, 64):
            with self.subTest(head=head):
                trip, placement = allocated(head=head)
                held = replayed(trip, placement)
                self.assertEqual({one.result for one in trip.operations},
                                 {place.operation.result for place in placement.places
                                  if place.operation.name != COMMIT})
                self.assertEqual(held, {region: leaving for region, (_, leaving) in trip.carried.items()})

    def test_a_carried_value_staging_through_a_slot_is_committed(self) -> None:
        """Asserts a maximum whose old value is still read lands in a slot and is copied over once it is not."""
        trip, placement = allocated(head=64)
        commits = [index for index, place in enumerate(placement.places) if place.operation.name == COMMIT]
        self.assertEqual(len(commits), 1)
        commit = placement.places[commits[0]]
        self.assertEqual(commit.into, "row_max")
        self.assertEqual(commit.operation.result, trip.carried["row_max"][1])
        entering = trip.carried["row_max"][0]
        reading = [index for index, place in enumerate(placement.places)
                   if entering in place.operation.operands]
        self.assertLess(max(reading), commits[0])

    def test_a_reduce_carries_no_value_through_the_slots_it_folds_through(self) -> None:
        """Asserts nothing a later operation reads sits in the two slots a row reduction writes over."""
        trip, placement = allocated(head=64)
        spent, holders = {"staged", "result"}, {}
        for index, place in enumerate(placement.places):
            if place.operation.name.startswith("tt.reduce."):
                for region, value in holders.items():
                    later = any(value in one.operation.operands for one in placement.places[index + 1:])
                    self.assertFalse(region in spent and later, f"{value} crosses a reduce in {region}")
            holders = {region: value for region, value in holders.items() if region != place.into}
            holders[place.into] = place.operation.result

    def test_the_slots_are_read_from_the_register_map(self) -> None:
        """Asserts the shared span is read as whole slots rather than as a count stated here."""
        trip, placement = allocated(head=64)
        registers = flash_register_map(queries=_TILE, head=64)
        named = {place.into for place in placement.places} - set(trip.carried)
        self.assertLessEqual(named, {"scores", "staged", "result", "partial"})
        self.assertEqual(registers.transient_span // registers.per_tile, 4)

    def test_a_span_the_phases_do_not_share_in_whole_tiles_is_refused(self) -> None:
        """Asserts a register map whose regions open inside a slot is refused rather than placed around."""
        trip = attention_trip(read_attention(), produces=0, consumes=1)
        wider = dataclasses.replace(flash_register_map(queries=_TILE, head=64), token_groups=2)
        with self.assertRaises(ProgramError) as raised:
            allocate(trip, registers=wider)
        self.assertIn("as a slot of the shared span", str(raised.exception))

    def test_a_scale_is_staged_into_a_slot_before_it_is_read(self) -> None:
        """Asserts a scale standing outside the loop is loaded into a slot rather than dropped."""
        trip = attention_trip(read_attention(flash_attention_scaled), produces=0, consumes=1)
        self.assertEqual(trip.constants, (0.125,))
        placement = allocate(trip, registers=flash_register_map(queries=_TILE, head=64))
        scaled = next(place for place in placement.places if place.operation.name == "arith.mulf"
                      and trip.values[place.operation.operands[1]].role == CONSTANT)
        self.assertIn(scaled.sources[1], {"scores", "staged", "result", "partial"})

    def test_the_launch_carries_the_steps_its_trip_placed(self) -> None:
        """Asserts a planned attention launch carries a body the emitter accepts rather than extents alone."""
        launch = planned_attention()
        self.assertIsInstance(launch, FlashLaunch)
        emit.check_steps(launch.body)
        emit.check_steps(launch.epilogue)
        products = [step for step in launch.body if isinstance(step, emit.Product)]
        self.assertEqual(len(products), launch.head // _TILE + launch.output_tiles)
        self.assertTrue(any(isinstance(step, emit.RowReduce) and step.kind == "vmax" for step in launch.body))
        self.assertTrue(any(isinstance(step, emit.RowReduce) and step.kind == "vadd" for step in launch.body))

    def test_the_rescale_of_the_output_reaches_every_tile(self) -> None:
        """Asserts the output is rescaled one tile at a time, some of its tiles lying in VMEM."""
        launch = planned_attention()
        rescales = [step for step in launch.body
                    if isinstance(step, emit.Accumulate) and step.kind == "vmul"]
        self.assertEqual(len(rescales), launch.output_tiles)
        self.assertEqual(sorted(step.tile for step in rescales), list(range(launch.output_tiles)))

    def test_a_scale_the_kernel_states_reaches_the_image(self) -> None:
        """Asserts scaling the scores stages a tile of the scale and leaves a different image."""
        plain = planned_attention()
        scaled = planned_attention(flash_attention_scaled)
        self.assertEqual((plain.constants, scaled.constants), ((), (0.125,)))
        self.assertNotEqual(compile_launch(plain)[0], compile_launch(scaled)[0])
        loads = [step for step in scaled.body if isinstance(step, emit.Load) and step.region == "constant"]
        self.assertEqual(len(loads), 1)

    def test_the_maximum_is_entered_at_what_the_kernel_states(self) -> None:
        """Asserts a maximum the kernel enters at no finite value is bounded rather than staged as it stands."""
        self.assertEqual(planned_attention().entering, -1e4)

    def test_a_launch_stating_its_extents_alone_is_refused(self) -> None:
        """Asserts an attention launch carrying no placed steps is refused rather than emitted empty."""
        with self.assertRaises(ProgramError) as raised:
            compile_launch(FlashLaunch(row_blocks=1, key_blocks=2, queries=_TILE, head=64))
        self.assertIn("carrying no steps", str(raised.exception))

    def test_a_trip_no_order_fits_is_refused(self) -> None:
        """Asserts a trip holding more values across a reduce than the slots surviving one is refused."""
        with self.assertRaises(ProgramError) as raised:
            allocate(crowded_trip(), registers=flash_register_map(queries=_TILE, head=64))
        self.assertIn("no order fits", str(raised.exception))


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
        self.assertTrue(launch.fuse_groups)
        self.assertEqual(planned.issued_over_useful, 1.0)
        self.assertEqual(planned.padding, ())
        self.assertEqual(planned.repack, ())

    def test_a_transform_ahead_of_the_matmul_activates_the_launch(self) -> None:
        """Asserts the transform the kernel writes ahead of its matmul is what sets the launch's activation.

        Every other field holds the plain product's value, since the transform rides the staging of the tile
        entering the array and changes neither the blocks the launch covers nor the drain it leaves.
        """
        plain, activated = compile_one(gemm)[1].launches[0], compile_one(gemm_activated)[1].launches[0]
        self.assertTrue(activated.activate)
        self.assertFalse(plain.activate)
        self.assertEqual(dataclasses.replace(activated, activate=False), plain)

    def test_a_transform_the_staging_cannot_run_is_refused(self) -> None:
        """Asserts a prologue naming operations the staging holds none for is refused rather than dropped."""
        with self.assertRaises(ProgramError):
            compile_one(gemm_unfusable_prologue)

    def test_the_feed_forward_layer_plans_as_a_plain_launch_and_an_activated_one(self) -> None:
        """Asserts a kernel writing both projections plans the GELU onto the launch consuming the intermediate."""
        matmuls, planned = compile_ffn()
        self.assertEqual(matmuls[1].left_source, 0)
        self.assertEqual([launch.activate for launch in planned.launches], [False, True])
        self.assertTrue(all(launch.fuse_groups for launch in planned.launches))

    def test_the_feed_forward_launches_meet_at_the_hidden_width(self) -> None:
        """Asserts the up projection's columns and the down projection's reduction cover the same width."""
        _, planned = compile_ffn()
        produced, consumed = planned.launches
        self.assertEqual(produced.column_tiles * _TILE, consumed.groups * consumed.loop_blocks * _BLOCK_DEPTH)

    def test_a_chained_matmul_reading_another_width_is_refused(self) -> None:
        """Asserts a product staged at a width its producer does not leave is refused rather than mispaired."""
        matmuls, _ = compile_ffn()
        narrowed = dataclasses.replace(matmuls[1], reduction=matmuls[1].reduction // 2)
        with self.assertRaises(ProgramError):
            plan((matmuls[0], narrowed), strides=dict(stride_am=192, stride_ak=1, stride_uk=192, stride_un=1,
                                                      stride_dk=16, stride_dn=1, stride_cm=16, stride_cn=1),
                 dims=({"m": 32}, {"m": 32, "n": 16}))

    def test_a_pair_one_image_holds_plans_as_one_launch(self) -> None:
        """Asserts a kernel at the shape one image runs plans the pair as a single fused launch."""
        launches = compile_fused().launches
        self.assertEqual(len(launches), 1)
        fused = launches[0]
        self.assertIsInstance(fused, FusedLaunch)
        self.assertEqual((fused.chunks, fused.steps, fused.produces, fused.consumes), (4, 8, 0, 1))

    def test_a_fused_launch_compiles_to_an_image(self) -> None:
        """Asserts the fused launch reaches an image and states the block that image drains."""
        image, staging = compile_launch(compile_fused().launches[0])
        self.assertGreater(len(image), 0)
        self.assertEqual((staging.rows, staging.columns), (_BLOCK_TOKENS, _TILE))

    def test_a_wider_hidden_chunk_plans_as_two_launches(self) -> None:
        """Asserts a pair whose chunk the registers cannot hold keeps the two launches it needs."""
        launches = compile_ffn()[1].launches
        self.assertEqual(len(launches), 2)
        self.assertFalse(any(isinstance(launch, FusedLaunch) for launch in launches))

    def test_a_single_matmul_is_the_same_plan_either_way(self) -> None:
        """Asserts the fusion flag leaves a kernel stating one matmul untouched, there being no pair to find."""
        matmuls = compile_one(gemm)[0]
        strides = dict(zip(_STRIDE_NAMES, (matmuls[0].reduction, 1, 48, 1, 48, 1), strict=True))
        self.assertEqual(plan(matmuls, strides=strides, fuse=True),
                         plan(matmuls, strides=strides, fuse=False))

    def test_a_transform_behind_the_matmul_is_refused(self) -> None:
        """Asserts a transform the kernel writes after its matmul is refused rather than left out of the image.

        The staging is the one place this machine runs a transform, and it holds the tile going in rather than
        the product coming out, so a launch is the whole of what a kernel wrote only when nothing follows its
        matmul. Planning one that activates nothing would emit the plain product and drop the transform.
        """
        with self.assertRaises(ProgramError):
            compile_one(gemm_gelu_epilogue)

    def test_the_drain_matches_the_partials_the_launch_leaves(self) -> None:
        """Asserts the drain covers one partial per block, which is what summing the groups on the device leaves."""
        _, planned = compile_one(gemm)
        launch, drain = planned.launches[0], planned.drains[0]
        self.assertEqual(launch.partials, 1)
        self.assertEqual(drain.shape, (launch.blocks, 1, launch.output_tiles, launch.block_m, _TILE))
        self.assertEqual(launch.blocks * launch.output_tiles, launch.column_tiles)

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

        # Covers a whole number of two tile blocks, so widening the block stays legal.
        base = dataclasses.replace(compile_one(gemm)[1].launches[0], column_tiles=4)
        reference, _ = compile_launch(base)
        for field, value in (("column_tiles", 8), ("groups", 6), ("loop_blocks", 4),
                             ("fuse_groups", False), ("block_m", 16), ("block_n", 32)):
            with self.subTest(field=field):
                altered, _ = compile_launch(dataclasses.replace(base, **{field: value}))
                self.assertNotEqual(altered, reference)

    def test_the_launch_count_stays_off_the_image(self) -> None:
        """Asserts `row_blocks` leaves the image alone, since it counts the launches a host issues."""

        base = dataclasses.replace(compile_one(gemm)[1].launches[0], column_tiles=4)
        self.assertEqual(compile_launch(dataclasses.replace(base, row_blocks=4))[0],
                         compile_launch(base)[0])

    def test_the_planned_drain_matches_the_one_the_image_writes(self) -> None:
        """Asserts the drain the plan states is the drain the emitter placed, for every block the planner produces.

        The board sizes its buffer and reshapes its partials from the plan's copy while the emitter settles its own,
        so a drift between the two reads a product out of the wrong rows rather than raising.
        """
        for kernel, dims in ((gemm, ()), (gemm_wide_columns, ({"m": 32},)), (gemm_short_rows, ())):
            with self.subTest(kernel=kernel.fn.__name__):
                _, planned = compile_one(kernel, dims=dims)
                launch, drain = planned.launches[0], planned.drains[0]
                _, staging = compile_launch(launch)
                self.assertEqual(drain.shape, (staging.blocks, staging.partials, staging.output_tiles,
                                               staging.rows, _TILE))
                self.assertEqual(drain.words, staging.drain_words)

    def test_a_spilling_block_plans_the_drain_its_image_writes(self) -> None:
        """Asserts a block accumulating in VMEM states the drain the emitter placed for it.

        A spilled tile drains once per VMEM bank, since consecutive blocks fold into opposite banks. A plan
        missing that doubling sizes the buffer at half what the image writes.
        """

        base = compile_one(gemm)[1].launches[0]
        launch = dataclasses.replace(base, column_tiles=4, block_n=4 * _TILE)
        self.assertTrue(register_map(block_m=launch.block_m, block_n=launch.block_n).spilled,
                        "the block this covers no longer spills, so it covers the resident path twice")
        _, staging = compile_launch(launch)
        self.assertEqual((launch.blocks, launch.partials, launch.output_tiles, launch.block_m, _TILE),
                         (staging.blocks, staging.partials, staging.output_tiles, staging.rows, _TILE))

    def test_a_launch_that_activates_is_refused(self) -> None:
        """Asserts a launch carrying an epilogue raises, since the emitter places the plain product only."""

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


@triton.jit
def flash_attention_kernel(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                           stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64,
                           D: tl.constexpr = 64, BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16):
    """Attends with the online softmax a flash attention kernel carries, one key block at a time."""
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
            scores = tl.dot(query, tl.trans(key))
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 accumulator / running_sum[:, None])


@triton.jit
def flash_attention_scaled(qp, kp, vp, op, stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                           stride_om, stride_od, M: tl.constexpr = 32, N: tl.constexpr = 64,
                           D: tl.constexpr = 64, BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16):
    """Attends as the kernel above does, scaling its scores the way attention over a head of this width does."""
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
            scores = tl.dot(query, tl.trans(key)) * 0.125
            block_max = tl.maximum(running_max, tl.max(scores, 1))
            probabilities = tl.exp(scores - block_max[:, None])
            alpha = tl.exp(running_max - block_max)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
            running_max = block_max
        tl.store(op + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 accumulator / running_sum[:, None])


@triton.jit
def gemm_folded_reduction(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                          M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 64,
                          BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies over a contraction one block covers, which canonicalization folds the reduction loop of."""
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
def gemm_square_tile(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     M: tl.constexpr = 64, N: tl.constexpr = 128, K: tl.constexpr = 192,
                     BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 64, BLOCK_K: tl.constexpr = 64):
    """Multiplies with an output block as wide as the reduction block is deep."""
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
def gemm_short_rows(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                    BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies with a token block covering one group rather than the two the token block holds."""
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
def gemm_activated(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                   BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Multiplies `gemm`'s product with the exact GELU standing between the left operand's load and the matmul."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                widened = tl.load(a).to(tl.float32)
                activated = (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(tl.bfloat16)
                accumulator += tl.dot(activated, tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_unfusable_prologue(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                            BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16,
                            BLOCK_K: tl.constexpr = 64):
    """Multiplies with a transform ahead of the matmul that the staging holds no operation for."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b = wp + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.exp(tl.load(a).to(tl.float32)).to(tl.bfloat16), tl.load(b))
                a += BLOCK_K * stride_ak
                b += BLOCK_K * stride_bk
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator)


@triton.jit
def gemm_gelu_epilogue(ap, wp, op, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       M: tl.constexpr = 64, N: tl.constexpr = 48, K: tl.constexpr = 192,
                       BLOCK_M: tl.constexpr = 32, BLOCK_N: tl.constexpr = 16, BLOCK_K: tl.constexpr = 64):
    """Stores the exact GELU of the product, which stands after the matmul rather than ahead of it."""
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
            activated = 0.5 * accumulator * (1.0 + tl.math.erf(accumulator / _ROOT_TWO))
            tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, activated)


@triton.jit
def ffn_chunked(ap, up, dp, op, stride_am, stride_ak, stride_uk, stride_un, stride_dk, stride_dn,
                stride_cm, stride_cn,
                M: tl.constexpr = 32, K: tl.constexpr = 192, H: tl.constexpr = 192, N: tl.constexpr = 16,
                BLOCK_M: tl.constexpr = 32, BLOCK_K: tl.constexpr = 64, BLOCK_H: tl.constexpr = 64,
                BLOCK_N: tl.constexpr = 16):
    """Runs both projections of a feed-forward layer, chunking the hidden width the GELU stands in the middle of."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for chunk in range(0, H, BLOCK_H):
        a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        u = up + offs_k[:, None] * stride_uk + (chunk + offs_h)[None, :] * stride_un
        hidden = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            hidden += tl.dot(tl.load(a), tl.load(u))
            a += BLOCK_K * stride_ak
            u += BLOCK_K * stride_uk
        widened = hidden.to(tl.bfloat16).to(tl.float32)
        activated = (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(tl.bfloat16)
        d = dp + (chunk + offs_h)[:, None] * stride_dk + offs_n[None, :] * stride_dn
        out += tl.dot(activated, tl.load(d))
    tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, out)


@triton.jit
def ffn_fused(ap, up, dp, op, stride_am, stride_ak, stride_uk, stride_un, stride_dk, stride_dn,
              stride_cm, stride_cn,
              M: tl.constexpr = 32, K: tl.constexpr = 128, H: tl.constexpr = 64, N: tl.constexpr = 16,
              BLOCK_M: tl.constexpr = 32, BLOCK_K: tl.constexpr = 64, BLOCK_H: tl.constexpr = 16,
              BLOCK_N: tl.constexpr = 16):
    """Runs both projections at the shape one image holds, the hidden width walked one tile per chunk."""
    offs_m, offs_k = tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_K)
    offs_h, offs_n = tl.arange(0, BLOCK_H), tl.arange(0, BLOCK_N)
    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for chunk in range(0, H, BLOCK_H):
        a = ap + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        u = up + offs_k[:, None] * stride_uk + (chunk + offs_h)[None, :] * stride_un
        hidden = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            hidden += tl.dot(tl.load(a), tl.load(u))
            a += BLOCK_K * stride_ak
            u += BLOCK_K * stride_uk
        widened = hidden.to(tl.bfloat16).to(tl.float32)
        activated = (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(tl.bfloat16)
        d = dp + (chunk + offs_h)[:, None] * stride_dk + offs_n[None, :] * stride_dn
        out += tl.dot(activated, tl.load(d))
    tl.store(op + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, out)


def compile_fused(*, fuse: bool = True):
    """Returns the plan the fused-shaped feed-forward kernel makes, with fusion on or off."""
    operands = {"ap": "*bf16", "up": "*bf16", "dp": "*bf16", "op": "*fp32"}
    constexprs = {parameter.name: parameter.default for parameter in ffn_fused.params
                  if parameter.is_constexpr}
    matmuls = read_program(ffn_fused, signature=signature_of(ffn_fused, operands=operands),
                           constexprs=constexprs)
    strides = dict(stride_am=128, stride_ak=1, stride_uk=64, stride_un=1, stride_dk=16, stride_dn=1,
                   stride_cm=16, stride_cn=1)
    return plan(matmuls, strides=strides, dims=({"m": 32}, {"m": 32, "n": 16}), fuse=fuse)


def compile_ffn():
    """Returns the matmuls the chunked feed-forward kernel states and the launches a plan makes of them."""
    operands = {"ap": "*bf16", "up": "*bf16", "dp": "*bf16", "op": "*fp32"}
    constexprs = {parameter.name: parameter.default for parameter in ffn_chunked.params
                  if parameter.is_constexpr}
    matmuls = read_program(ffn_chunked, signature=signature_of(ffn_chunked, operands=operands),
                           constexprs=constexprs)
    strides = dict(stride_am=192, stride_ak=1, stride_uk=192, stride_un=1, stride_dk=16, stride_dn=1,
                   stride_cm=16, stride_cn=1)
    return matmuls, plan(matmuls, strides=strides, dims=({"m": 32}, {"m": 32, "n": 16}))


def read_attention(kernel=flash_attention_kernel):
    """Returns the matmuls one attention kernel states.

    Args:
        kernel: The attention kernel to read.

    Returns:
        The score product and the product consuming it.
    """
    constexprs = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    return read_program(kernel, signature=signature_of(kernel, operands=_ATTENTION_OPERANDS),
                        constexprs=constexprs)


def planned_attention(kernel=flash_attention_kernel):
    """Returns the launch one attention kernel plans to, its strides bound as one head's rows.

    Args:
        kernel: The attention kernel to plan.

    Returns:
        The launch.
    """
    walked = {"m": 64, "n": 64, "d": 1}
    strides = {f"stride_{name}{axis}": walked[axis]
               for name in ("q", "k", "v", "o") for axis in ("m", "n", "d") if f"{name}{axis}" != "qn"}
    return plan(read_attention(kernel), strides=strides).launches[0]


def allocated(*, head: int):
    """Returns the attention kernel's trip and the placement one head width allocates it under.

    Args:
        head: Width of the head the block attends over.

    Returns:
        The trip and its placement.
    """
    trip = attention_trip(read_attention(), produces=0, consumes=1)
    return trip, allocate(trip, registers=flash_register_map(queries=_TILE, head=head))


def replayed(trip, placement):
    """Returns the value every region holds once the placement has run, refusing a read nothing answers.

    Args:
        trip: The dataflow the placement was made of.
        placement: The placement to replay.

    Returns:
        The value each carried region holds at the end.

    Raises:
        AssertionError: If an operation reads a region holding another value than the operand it names.
    """
    held = {region: entering for region, (entering, _) in trip.carried.items()}
    for place in placement.places:
        for operand, source in zip(place.operation.operands, place.sources, strict=True):
            if trip.values[operand].role in (STREAMED, CONSTANT):
                continue
            if held.get(source) != operand:
                message = (f"{place.operation.name} at {place.operation.position} reads {source}, which holds "
                           f"{held.get(source)} where it names {operand}")
                raise AssertionError(message)
        held[place.into] = place.operation.result
    return {region: held[region] for region in trip.carried}


def crowded_trip():
    """Returns a trip holding more values across a reduce than the slots surviving one carry.

    Returns:
        The trip.
    """
    carried = {"row_max": (100, 101), "row_sum": (200, 201), "accumulator": (300, 301)}
    # Each value is computed before the fold and read after it, so no order defers one past the fold.
    spread = [Operation(name="arith.addf", position=index, result=10 + index,
                        operands=(100 if index == 0 else 9 + index, 100)) for index in range(5)]
    folded = Operation(name="tt.reduce.arith.addf", position=5, result=20, operands=(14,))
    chained = [Operation(name="arith.addf", position=6 + index, result=21 + index,
                         operands=(10 + index, 20 + index)) for index in range(3)]
    leaving = (Operation(name="arith.addf", position=9, result=101, operands=(13, 23)),
               Operation(name="arith.addf", position=10, result=201, operands=(200, 101)),
               Operation(name="arith.mulf", position=11, result=301, operands=(300, 101)))
    operations = (*spread, folded, *chained, *leaving)
    values = {identifier: Value(identifier, CARRIED, region)
              for region, pair in carried.items() for identifier in pair}
    for operation in operations:
        values.setdefault(operation.result, Value(operation.result, TRANSIENT, None))
    return Trip(operations=operations, after=(), values=values, carried=carried)


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
