"""Contains tests for the GEMM emitter, holding it to the hand-placed image it derives its schedule from.

The emitter exists to replace a hand-tuned bundle stream with one placed from the ISA table, so the only claim worth
testing is that the two agree word for word.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as the package ships its modules beside the tests rather
# than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

import asm  # noqa: E402
import bf16  # noqa: E402
import emit  # noqa: E402
import kernels  # noqa: E402
import numpy as np  # noqa: E402
from compiler_tests.test_layout import renaming_onto  # noqa: E402
from layout import register_map  # noqa: E402

_ACTIVATION_POINTER: int = 0
"""Scalar register the activation operand's host address lives in, as `emit.gemm_operands` assigns it."""

_F_LBEGIN: int = 1
"""F-slot opcode opening a hardware loop, which `AsmBuilder.lbegin` encodes."""

_F_LOOP_END: int = 2
"""F-slot opcode closing the innermost hardware loop."""

_E_OPCODE_SHIFT: int = 15
"""Bit the opcode starts at within an E-slot word, above its three five bit register fields."""


class EmitterSerialisationTest(unittest.TestCase):
    """Covers the emitted stream as the image bytes a compiler backend writes out."""

    def test_the_emitted_stream_serialises_to_the_hand_written_image(self) -> None:
        """Asserts the emitted stream, wrapped in a kernel, writes the bytes the hand-written builder does."""
        for activate in (False, True):
            for column_tiles in (1, 3, 6):
                for groups in (1, 3):
                    for loop_blocks in (2, 4, 6):
                        with self.subTest(activate=activate, column_tiles=column_tiles,
                                          groups=groups, loop_blocks=loop_blocks):
                            hand = kernels.build_looped_gemm(
                                column_tiles, groups, activate=activate,
                                loop_blocks=loop_blocks).image.assemble(pad_to=4)
                            derived = asm.AsmKernel(
                                name="looped_gemm",
                                bundles=emit.looped_gemm_stream(
                                    column_tiles=column_tiles, groups=groups, loop_blocks=loop_blocks,
                                    activate=activate)[0]).assemble(pad_to=4)
                            self.assertIsNotNone(renaming_onto(derived, hand))
                            # Holds the E-slot opcodes apart, which `renaming_onto` does not compare, so a step
                            # dropping or misplacing its vgelu fails here.
                            self.assertEqual(tuple(_e_opcode(word) for word in derived),
                                             tuple(_e_opcode(word) for word in hand))


class EmitterRegisterBudgetTest(unittest.TestCase):
    """Holds the emitter's register use to the budget `layout.register_map` derives for the same block."""

    def test_the_emitted_image_touches_the_registers_the_map_lays_out(self) -> None:
        """Asserts the emitted GEMM names every register the map lays out for its block, and no other.

        The emitter takes its bases from the map, so the two agree on which registers a block uses rather than only
        on how many. A base drifting from the map would leave a register named outside its region.
        """
        words, _ = emit.looped_gemm_stream(column_tiles=bf16._STEP_TILES, groups=bf16._TOKEN_GROUPS,
                                           loop_blocks=bf16._LOOP_BLOCKS)
        touched: set[int] = set()
        for word in words:
            decoded = asm._decode(word)
            touched |= set(decoded["reads"]) | {register for register, _ in decoded["writes"]}
        mapped = register_map(block_m=bf16._BLOCK_TOKENS, block_n=bf16._TILE)
        per_tile = mapped.token_groups * bf16._TILE_VREGS
        laid_out: set[int] = set()
        for base, width in ((mapped.accumulator, mapped.resident * per_tile), (mapped.activation, per_tile),
                            (mapped.weight, bf16._TILE_VREGS), (mapped.result, bf16._TILE_VREGS)):
            laid_out |= set(range(base, base + width))
        self.assertEqual(touched, laid_out)


class EmitterHandoverTest(unittest.TestCase):
    """Holds what the emitter tells a host to stage against what the image it placed actually moves."""

    def test_the_staging_drains_the_rows_the_image_stores(self) -> None:
        """Asserts the drain rows a host is told to expect are the rows the image's store descriptors move.

        A host stages and reduces from these figures, so one that drifts from the image leaves the two halves
        describing different launches with nothing between them to disagree.
        """
        for counts in (dict(column_tiles=3, groups=3, loop_blocks=2),
                       dict(column_tiles=4, groups=3, loop_blocks=2, block_n=32),
                       dict(column_tiles=3, groups=3, loop_blocks=2, block_m=16)):
            with self.subTest(**counts):
                words, staging = emit.looped_gemm_stream(**counts)
                stored = {_descriptor_rows(word) for word in words if _stores(word)}
                self.assertEqual(stored, {staging.drain_rows})

    def test_the_activation_slack_covers_the_blocks_the_image_prefetches(self) -> None:
        """Asserts the slack a host allocates is the activation the image's trailing prefetch reads past the end.

        The block loop stages one block ahead, so the last iteration reads a block that is not there. A slack taken
        from a constant rather than from the block the image stages leaves that read off the end of the buffer.
        """
        for counts in (dict(column_tiles=3, groups=3, loop_blocks=2),
                       dict(column_tiles=4, groups=3, loop_blocks=2, block_n=32),
                       dict(column_tiles=3, groups=3, loop_blocks=2, block_m=16)):
            with self.subTest(**counts):
                words, staging = emit.looped_gemm_stream(**counts)
                staged = {_descriptor_rows(word) for word in words
                          if _loads(word) and _descriptor_base(word) == _ACTIVATION_POINTER}
                self.assertEqual(staged, {staging.rows})
                self.assertEqual(staging.activation_slack, bf16._STEPS * staging.rows * bf16._ROW_BYTES)

    def test_the_drain_fires_in_the_order_the_reduction_reads_it(self) -> None:
        """Asserts the loops the image drains under run in the order `reduce_drain`'s leading axes assume.

        The rows a descriptor moves say nothing about the order the loops issue them in, so a reader whose axes are
        permuted still sees a buffer of the right size and sums the wrong rows together.
        """
        for counts in (dict(column_tiles=3, groups=3, loop_blocks=2),
                       dict(column_tiles=4, groups=3, loop_blocks=2, block_n=32),
                       dict(column_tiles=3, groups=3, loop_blocks=2, block_m=16)):
            with self.subTest(**counts):
                words, staging = emit.looped_gemm_stream(**counts)
                # Pads each vector, the drain sitting outside the group loop and carrying the block alone.
                issued = [(counters + (0,))[:2] for counters in _drain_sequence(words)]
                self.assertEqual(issued, [(block, partial) for block in range(staging.blocks)
                                          for partial in range(staging.partials)])

    def test_the_drain_reduces_to_the_product_the_launch_covers(self) -> None:
        """Asserts `reduce_drain` recovers each column's value from a buffer laid out as the image fills it."""
        for counts in (dict(column_tiles=4, groups=3, loop_blocks=2, block_n=32),
                       dict(column_tiles=3, groups=3, loop_blocks=2, block_m=16)):
            with self.subTest(**counts):
                _, staging = emit.looped_gemm_stream(**counts)
                shape = (staging.blocks, staging.partials, staging.output_tiles, staging.rows, bf16._TILE)
                wanted = np.arange(staging.rows * staging.columns, dtype=np.float32).reshape(
                    staging.rows, staging.columns)
                drained = np.empty(shape, dtype=np.float32)
                for block, partial, tile in np.ndindex(*shape[:3]):
                    column = (block * staging.output_tiles + tile) * bf16._TILE
                    # Splits each column across its partials, so a reduction summing the wrong axis misses the value.
                    drained[block, partial, tile] = wanted[:, column : column + bf16._TILE] / staging.partials
                np.testing.assert_allclose(bf16.reduce_drain(drained.reshape(-1), shape=shape), wanted, rtol=1e-6)


class SpillReloadTest(unittest.TestCase):
    """Covers where a spilled tile's reload lands in the step that folds it."""

    def test_a_hoisted_reload_issues_before_the_first_push(self) -> None:
        """Asserts a block owning a scratch tile per group loads its accumulators at the step's head.

        The reload reads what earlier steps left, so placing it ahead of the array work hides its latency behind
        the pushes instead of stalling the adds that consume it.
        """
        registers = register_map(block_m=32, block_n=64)
        self.assertTrue(registers.reload_hoisted)
        rows = emit.vmem_map(output_tiles=registers.output_tiles, token_groups=registers.token_groups,
                             spilled=registers.spilled)
        step = emit.gemm_step(activation_row=rows["activation"][0], weight_row=rows["weight"][0],
                              close_loop=False, registers=registers, tile=0, spill_row=rows["spill"][0])
        first_push = min(offset for offset, slots in step.items() if "m" in slots and offset > emit._MATLOAD_AT)
        loaded = {register
                  for offset, slots in step.items() if offset < first_push and "x" in slots
                  for register, _ in asm._decode(slots["x"])["writes"]}
        scratch = set(range(registers.scratch, registers.scratch + registers.scratch_tiles * bf16._TILE_VREGS))
        self.assertTrue(scratch <= loaded, "a scratch tile is left unloaded when the pushes begin")


class PointerAdvanceTest(unittest.TestCase):
    """Covers the adds the emitter moves an operand's pointer with."""

    def test_a_wide_displacement_splits(self) -> None:
        """Asserts the adds a displacement is split across carry it whole, each inside the immediate's reach."""
        bundles: dict[int, dict[str, int]] = {}

        def put(*, offset: int, **slots: int) -> None:
            """Records the bundle one add is placed at."""
            bundles[offset] = slots

        for delta in (512, -48640, emit._MAX_SADDI, emit._MAX_SADDI + 1, -3 * emit._MAX_SADDI):
            with self.subTest(delta=delta):
                bundles.clear()
                at = emit._emit_split_advance(put=put, builder=asm.AsmBuilder(), at=0, register=2, delta=delta,
                                              gap=1)
                addends = [_scalar_addend(slots["s"]) for slots in bundles.values()]
                self.assertEqual(sum(addends), delta)
                self.assertEqual(at, len(addends))
                self.assertTrue(all(abs(addend) <= emit._MAX_SADDI for addend in addends))

    def test_a_wide_rewind_still_emits(self) -> None:
        """Asserts a block wide and deep enough to rewind past one immediate still places every bundle."""
        words, staging = emit.looped_gemm_stream(column_tiles=48, groups=4, loop_blocks=4, block_m=32, block_n=64)
        self.assertEqual(staging.output_tiles, 4)
        addends = [_scalar_addend(word) for word in words if asm._decode(word)["s_writes"]]
        self.assertIn(-emit._MAX_SADDI, addends)
        # Scheduling refuses a stream it cannot space, so one it returns is legal.
        self.assertTrue(asm.schedule(list(words)))


class BundleMergeTest(unittest.TestCase):
    """Covers the merge every placed bundle goes through, which is the only thing holding one word per slot."""

    def test_a_slot_the_offset_already_carries_is_refused(self) -> None:
        """Asserts a second word of one slot raises instead of replacing the word already placed there.

        A bundle issues one word per slot, so replacing the first would drop an instruction from an image nothing
        downstream re-checks, which reaches a board as wrong numbers.
        """
        builder = asm.AsmBuilder()
        bundles: dict[int, dict[str, int]] = {}
        emit._merge(bundles=bundles, offset=0, slots={"x": builder.vld(vreg=16, row=48)})
        with self.assertRaises(ValueError):
            emit._merge(bundles=bundles, offset=0, slots={"x": builder.vst(vreg=24, row=208)})
        self.assertEqual(bundles[0]["x"], builder.vld(vreg=16, row=48), "the placed load was replaced")

    def test_a_slot_the_offset_leaves_free_co_issues(self) -> None:
        """Asserts a merge naming a free slot lands beside what the offset holds, which is what co-issue is."""
        builder = asm.AsmBuilder()
        bundles: dict[int, dict[str, int]] = {}
        emit._merge(bundles=bundles, offset=0, slots={"m": builder.vmatpush(vreg=8)})
        emit._merge(bundles=bundles, offset=0, slots={"x": builder.vld(vreg=16, row=48)})
        self.assertEqual(sorted(bundles[0]), ["m", "x"])

    def test_a_descriptor_co_issued_with_a_compute_slot_is_refused(self) -> None:
        """Asserts a compute slot merged onto a descriptor raises, since the two encodings overlap."""
        builder = asm.AsmBuilder()
        bundles: dict[int, dict[str, int]] = {}
        emit._merge(bundles=bundles, offset=0, slots={"d": builder.dma_load(
            slot=0, vmem_address=0, rows=16, base_sreg=1, stride_sreg=3)})
        with self.assertRaises(ValueError):
            emit._merge(bundles=bundles, offset=0, slots={"x": builder.vld(vreg=16, row=48)})

    def test_the_emitted_step_places_one_word_per_slot(self) -> None:
        """Asserts a step's own placement never asks one bundle to issue two words of a slot."""
        step = emit.gemm_step(activation_row=0, weight_row=0, close_loop=False)
        self.assertTrue(all(slots for slots in step.values()))


def _drain_sequence(words: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Returns the enclosing loop counters at every store descriptor, in the order the image issues them.

    The hardware loops are unrolled here rather than measured on a device, so the order a drain lands in the buffer
    is read off the image itself.
    """
    events: list[tuple] = []
    for word in words:
        if _stores(word):
            events.append(("store",))
        field = (word >> asm._F_SHIFT) & ((1 << asm.F_SLOT_W) - 1)
        operation = (field >> (asm.F_SLOT_W - 3)) & 0x7 if field else 0
        if operation == _F_LBEGIN:
            events.append(("lbegin", field & 0xFFFF))
        elif operation == _F_LOOP_END:
            events.append(("loop_end",))

    issued: list[tuple[int, ...]] = []

    def walk(start: int, counters: tuple[int, ...]) -> int:
        """Runs the events from one position, returning the position past the loop it closes."""
        at = start
        while at < len(events):
            event = events[at]
            if event[0] == "lbegin":
                for trip in range(event[1]):
                    past = walk(at + 1, (*counters, trip))
                at = past
            elif event[0] == "loop_end":
                return at + 1
            else:
                issued.append(counters)
                at += 1
        return at

    walk(0, ())
    return issued


def _stores(word: int) -> bool:
    """Returns whether one bundle carries a descriptor writing host memory."""
    return bool(asm._decode(word)["dma"]) and bool((word >> asm._D_SHIFT) >> 35 & 1)


def _loads(word: int) -> bool:
    """Returns whether one bundle carries a descriptor reading host memory."""
    return bool(asm._decode(word)["dma"]) and not (word >> asm._D_SHIFT) >> 35 & 1


def _descriptor_base(word: int) -> int:
    """Returns the scalar register one descriptor reads its host base address from."""
    return (word >> asm._D_SHIFT) >> 2 & 0x3


def _descriptor_rows(word: int) -> int:
    """Returns the rows one descriptor moves, which it holds one less than."""
    return ((word >> asm._D_SHIFT) >> 4 & 0x7F) + 1


def _scalar_addend(word: int) -> int:
    """Returns the signed immediate one scalar add carries."""
    immediate = (word >> asm._S_SHIFT) & 0xFFFF
    return immediate - 0x10000 if immediate & 0x8000 else immediate


def _loop_opcode(word: int) -> tuple[int, int]:
    """Returns the F-slot loop opcode one bundle carries and the trip count it names."""
    field = (word >> asm._F_SHIFT) & ((1 << asm.F_SLOT_W) - 1)
    return ((field >> (asm.F_SLOT_W - 3)) & 0x7 if field else 0), field & 0xFFFF


def _e_opcode(word: int) -> int:
    """Returns the E-slot opcode one bundle carries, or zero for an empty slot."""
    return ((word >> asm._E_SHIFT) & ((1 << asm.E_SLOT_W) - 1)) >> _E_OPCODE_SHIFT


def _chunk_body(words: tuple[int, ...], *, chunks: int) -> tuple[int, ...]:
    """Returns the words the fused layer's chunk loop encloses, which is where its weight staging rides."""
    depth, first = 0, None
    for at, word in enumerate(words):
        operation, trips = _loop_opcode(word)
        if operation == _F_LBEGIN and (first is not None or trips == chunks):
            depth += 1
            first = at + 1 if first is None else first
        elif operation == _F_LOOP_END and first is not None:
            depth -= 1
            if not depth:
                return words[first:at]
    return ()


class FusedFeedForwardEmitterTest(unittest.TestCase):
    """Holds the emitted fused layer to the staging its sweeps carry."""

    def test_the_weight_staging_rides_the_sweeps(self) -> None:
        """Asserts a sweep never opens with more descriptors outstanding than the slots carrying them."""
        for steps, output_tiles in ((32, 32), (48, 48)):
            with self.subTest(steps=steps, output_tiles=output_tiles):
                words, _ = emit.fused_ffn_stream(chunks=2, steps=steps, output_tiles=output_tiles)
                outstanding = 0
                for word in _chunk_body(words, chunks=2):
                    if _loads(word):
                        outstanding += 1
                        self.assertLessEqual(outstanding, asm.DMA_SLOT_COUNT)
                    elif _loop_opcode(word)[0] == _F_LBEGIN:
                        outstanding = 0

    def test_the_chunk_stages_exactly_its_weights(self) -> None:
        """Asserts a chunk's descriptors move its own two weight tiles and no more."""
        for steps, output_tiles in ((32, 32), (48, 48)):
            with self.subTest(steps=steps, output_tiles=output_tiles):
                words, _ = emit.fused_ffn_stream(chunks=2, steps=steps, output_tiles=output_tiles)
                moved = sum(_descriptor_rows(word) for word in _chunk_body(words, chunks=2)
                            if _loads(word))
                self.assertEqual(moved, (steps + output_tiles) * bf16._TILE)

    def test_the_splits_cover_every_tile_of_a_sweep(self) -> None:
        """Asserts splitting a sweep for its issue points leaves its trip count whole."""
        for steps, output_tiles in ((32, 32), (48, 48)):
            with self.subTest(steps=steps, output_tiles=output_tiles):
                words, _ = emit.fused_ffn_stream(chunks=2, steps=steps, output_tiles=output_tiles)
                trips = [named for word in _chunk_body(words, chunks=2)
                         for opcode, named in (_loop_opcode(word),) if opcode == _F_LBEGIN]
                self.assertEqual(sum(trips), steps + output_tiles)

    def test_the_emitted_layer_states_the_block_it_drains(self) -> None:
        """Asserts the staging names the token block and the width the layer's output covers."""
        _, staging = emit.fused_ffn_stream(chunks=2, steps=2, output_tiles=4)
        self.assertEqual((staging.rows, staging.columns), (bf16._BLOCK_TOKENS, 4 * bf16._TILE))
        self.assertEqual(staging.drain_rows, bf16._BLOCK_TOKENS * 4)



class StepBodyTest(unittest.TestCase):
    """Tests the checks a body of steps is placed under."""

    def test_a_body_leaving_the_spent_regions_alone_is_accepted(self) -> None:
        """Verifies that a body reading nothing a reduction folded through states a surviving order."""
        emit.check_steps((emit.Load(into="scores", region="query"),
                          emit.RowReduce(kind="vmax", source="scores", into="partial"),
                          emit.Combine(kind="vadd", into="row_sum", left="row_sum", right="partial")))

    def test_a_read_of_a_region_a_reduction_spent_is_refused(self) -> None:
        """Verifies that reading the staging a reduction transposed through is refused rather than emitted."""
        with self.assertRaises(ValueError):
            emit.check_steps((emit.Load(into="staged", region="query"),
                              emit.RowReduce(kind="vmax", source="scores", into="partial"),
                              emit.Combine(kind="vmul", into="row_sum", left="row_sum", right="staged")))

    def test_every_step_states_the_registers_it_uses(self) -> None:
        """Verifies that every step the vocabulary carries reports the register regions it touches."""
        for step in (emit.Load(into="scores", region="query"), emit.Store(source="scores", region="output"),
                     emit.Product(pushed="scores", weight="staged"),
                     emit.Combine(kind="vadd", into="partial", left="partial", right="result"),
                     emit.Transform(kind="vexp", into="scores", source="scores"),
                     emit.RowReduce(kind="vmax", source="scores", into="partial"),
                     emit.Accumulate(kind="vmul", tile=0, operand="staged")):
            reads, writes, spent = emit.classify_register_use(step)
            self.assertIsInstance(reads | writes | spent, frozenset)


if __name__ == "__main__":
    unittest.main()
