"""Contains tests for the physical encodings and the hazard pricing performed by the public assembler.

Mis-spacing is silent on a machine with no interlocks, so the distances asserted here track compiler/minitpuisa.td,
which the RTL's own WB_W_* constants are held against upstream.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Extends the import path before the assembler imports below resolve, as the package ships its modules beside the
# tests rather than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))

import asm  # noqa: E402
from asm import AsmBuilder, _delay_of, mxu_output_fifo, schedule  # noqa: E402


class AssemblerEncodingTest(unittest.TestCase):
    """Verifies the physical E-slot encodings the assembler emits."""

    def test_layernorm_e_slot_encodings(self) -> None:
        """Verifies the opcode and register fields of every E-slot operation layernorm uses."""
        builder = AsmBuilder()
        self.assertEqual(_e_fields(builder.vadd(vd=3, va=4, vb=5)), (1, 3, 4, 5))
        self.assertEqual(_e_fields(builder.vsub(vd=6, va=7, vb=8)), (2, 6, 7, 8))
        self.assertEqual(_e_fields(builder.vmul(vd=9, va=10, vb=11)), (3, 9, 10, 11))
        self.assertEqual(_e_fields(builder.vrsqrt(vd=12, va=13)), (8, 12, 13, 0))
        self.assertEqual(_e_fields(builder.vredsum(vd=14, va=15)), (9, 14, 15, 0))
        self.assertEqual(_e_fields(builder.vredmax(vd=16, va=17)), (10, 16, 17, 0))


class SchedulingTest(unittest.TestCase):
    """Verifies the delays the scheduler inserts between a producer and its consumers."""

    def test_load_to_consumer_is_spaced_to_L7(self) -> None:
        """Verifies that a consumer of a loaded VREG is spaced to the load's writeback distance."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(x=builder.vld(vreg=0, row=0))
            builder.bundle(e=builder.vadd(vd=1, va=0, vb=0))
            builder.bundle(f=builder.halt())

        words = _image(body)
        self.assertEqual(asm._delay_of(words[0]), 6)

    def test_independent_bundles_need_no_delay(self) -> None:
        """Verifies that two loads writing different VREGs issue back to back."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(x=builder.vld(vreg=0, row=0))
            builder.bundle(x=builder.vld(vreg=1, row=1))
            builder.bundle(f=builder.halt())

        words = _image(body)
        self.assertEqual(asm._delay_of(words[0]), 0)

    def test_write_port_collision_is_separated(self) -> None:
        """Verifies that an ALU writeback is moved off the cycle a load's writeback occupies."""
        def body(builder: AsmBuilder) -> None:
            # Lands at +6, which is where the following ALU operation would land unaided.
            builder.bundle(x=builder.vld(vreg=0, row=0))
            builder.bundle(e=builder.vadd(vd=2, va=3, vb=4))
            builder.bundle(f=builder.halt())

        words = _image(body)
        self.assertGreaterEqual(asm._delay_of(words[0]), 1)

    def test_vmatpush_gap_exceeds_one_delay_field_and_gets_a_filler(self) -> None:
        """Verifies that a gap wider than the 63-cycle delay field is covered by a filler bundle."""
        def body(builder: AsmBuilder) -> None:
            # Occupies the array for 83 cycles, which one delay field cannot express.
            builder.bundle(m=builder.vmatpush(vreg=0))
            builder.bundle(m=builder.vmatpop(vreg=1))
            builder.bundle(f=builder.halt())

        words = _image(body)
        self.assertGreater(len(words), 3, "expected a filler bundle to be inserted")


class PortAExclusivityTest(unittest.TestCase):
    """Verifies that VREG read port A is claimed by one reader per bundle.

    No delay field separates two consumers inside one bundle, so co-issue is rejected at assembly time rather than
    reading the wrong register on silicon.
    """

    def test_sfu_co_issued_with_vmatpush_is_rejected(self) -> None:
        """Verifies that an SFU operation co-issued with a vmatpush raises."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(m=builder.vmatpush(vreg=12), e=builder.vgelu(vd=16, va=16))
            builder.bundle(f=builder.halt())

        with self.assertRaisesRegex(ValueError, "read VREG port A"):
            _image(body)

    def test_vst_co_issued_with_vmatload_is_rejected(self) -> None:
        """Verifies that a store co-issued with a vmatload raises."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(m=builder.vmatload(vreg=24), x=builder.vst(vreg=0, row=0))
            builder.bundle(f=builder.halt())

        with self.assertRaisesRegex(ValueError, "read VREG port A"):
            _image(body)

    def test_vld_and_vmatpop_are_not_port_a_readers(self) -> None:
        """Verifies that a VMEM load and a vmatpop co-issue with matrix operations."""
        def body(builder: AsmBuilder) -> None:
            # vld reads VMEM and vmatpop only writes, so neither claims port A.
            builder.bundle(m=builder.vmatpush(vreg=12), x=builder.vld(vreg=0, row=0))
            builder.bundle(m=builder.vmatpop(vreg=28), e=builder.vadd(vd=1, va=2, vb=3))
            builder.bundle(f=builder.halt())

        self.assertGreaterEqual(len(_image(body)), 3)


class MatrixFifoTest(unittest.TestCase):
    """Verifies that a push burst is held within the MXU's per-lane output FIFO.

    The array's 82-cycle latency leaves the draining pops unrun, so a burst past the depth drops results silently and
    no delay repairs it.
    """

    def test_one_push_past_capacity_is_rejected(self) -> None:
        """Verifies that a burst one push wider than the FIFO raises."""
        pushes = _fifo_capacity() + 1

        def body(builder: AsmBuilder) -> None:
            for vreg in range(pushes):
                builder.bundle(m=builder.vmatpush(vreg=12 + vreg))
            builder.bundle(f=builder.halt())

        with self.assertRaisesRegex(ValueError, "overflows the MXU output FIFO"):
            _image(body)

    def test_capacity_pushes_fill_it_exactly(self) -> None:
        """Verifies that a burst exactly the depth of the FIFO is accepted."""
        pushes = _fifo_capacity()

        def body(builder: AsmBuilder) -> None:
            for vreg in range(pushes):
                builder.bundle(m=builder.vmatpush(vreg=12 + vreg))
            for vreg in range(pushes):
                builder.bundle(m=builder.vmatpop(vreg=vreg))
            builder.bundle(f=builder.halt())

        self.assertGreaterEqual(len(_image(body)), 2 * pushes + 1)

    def test_draining_between_bursts_lets_more_pushes_through(self) -> None:
        """Verifies that pops between bursts restore the capacity for a second full burst."""
        pushes = _fifo_capacity()

        def body(builder: AsmBuilder) -> None:
            for _ in range(2):
                for vreg in range(pushes):
                    builder.bundle(m=builder.vmatpush(vreg=12 + vreg))
                for vreg in range(pushes):
                    builder.bundle(m=builder.vmatpop(vreg=vreg))
            builder.bundle(f=builder.halt())

        self.assertGreaterEqual(len(_image(body)), 4 * pushes + 1)


class FlushSlotTest(unittest.TestCase):
    """Verifies that a flush waits only on DMA slots the stream issues a descriptor on.

    Flushing an unissued slot blocks forever and wedges the device. Slots are counted over the whole stream, because
    a loop body's flush legitimately collects a descriptor issued in the previous iteration.
    """

    def test_flushing_a_slot_never_issued_on_is_rejected(self) -> None:
        """Verifies that a flush mask covering an unissued slot raises."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(d=builder.dma_load(slot=0, vmem_address=0, rows=4, base_sreg=0, stride_sreg=3))
            builder.bundle(f=builder.flush_slot(mask=3))
            builder.bundle(f=builder.halt())

        with self.assertRaisesRegex(ValueError, "waits on DMA slot"):
            _image(body)

    def test_flushing_only_the_issued_slot_is_accepted(self) -> None:
        """Verifies that a flush mask covering the issued slot alone is accepted."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(d=builder.dma_load(slot=0, vmem_address=0, rows=4, base_sreg=0, stride_sreg=3))
            builder.bundle(f=builder.flush_slot(mask=1))
            builder.bundle(f=builder.halt())

        self.assertTrue(_image(body))

    def test_both_slots_issued_lets_the_pair_flush(self) -> None:
        """Verifies that a two-slot flush mask is accepted once both slots carry a descriptor."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(d=builder.dma_load(slot=0, vmem_address=0, rows=4, base_sreg=0, stride_sreg=3))
            builder.bundle(d=builder.dma_load(slot=1, vmem_address=8, rows=4, base_sreg=2, stride_sreg=3))
            builder.bundle(f=builder.flush_slot(mask=3))
            builder.bundle(f=builder.halt())

        self.assertTrue(_image(body))

    def test_a_descriptor_issued_after_the_flush_still_counts(self) -> None:
        """Verifies that a descriptor issued later in the stream satisfies an earlier flush."""
        def body(builder: AsmBuilder) -> None:
            builder.bundle(d=builder.dma_load(slot=0, vmem_address=0, rows=4, base_sreg=0, stride_sreg=3))
            builder.bundle(f=builder.flush_slot(mask=3))
            builder.bundle(d=builder.dma_load(slot=1, vmem_address=8, rows=4, base_sreg=2, stride_sreg=3))
            builder.bundle(f=builder.halt())

        self.assertTrue(_image(body))


class VmemAccessCostTest(unittest.TestCase):
    """A VMEM access holds the compute port for exactly one cycle.

    The sublanes read consecutive rows, so their banks are a permutation and no access splits.
    """

    def test_a_load_does_not_delay_the_next_access(self) -> None:
        """One access clears the compute port before the next issues."""
        builder = AsmBuilder()
        builder.bundle(x=builder.vld(vreg=0, row=0))
        builder.bundle(x=builder.vld(vreg=1, row=64))
        builder.bundle(f=builder.halt())
        words = schedule(list(builder.bundles))
        self.assertEqual(_delay_of(words[0]), 0)

    def test_every_access_costs_one_cycle(self) -> None:
        """The decoder prices each access at one cycle whichever sublanes it names."""
        builder = AsmBuilder()
        for mask in (0xF, 0b0101, 0b0001):
            self.assertEqual(asm._decode(builder.vld(vreg=0, row=0, mask=mask))["vmem_cycles"], 1)


class RememberedScheduleTest(unittest.TestCase):
    """Verifies that the cross-process schedule memo returns an identical image or misses.

    A remembered schedule is launched without being re-derived, so a stale or corrupt image reaches the board.

    Attributes:
        directory: The temporary directory backing the memo for one test.
        previous_cache: The MINITPU_SCHEDULE_CACHE value the environment carried before the test.
    """

    def setUp(self) -> None:
        """Points the memo at a temporary directory and clears the in-process cache."""
        self.directory = pathlib.Path(tempfile.mkdtemp())
        self.previous_cache = os.environ.get("MINITPU_SCHEDULE_CACHE")
        os.environ["MINITPU_SCHEDULE_CACHE"] = str(self.directory)
        asm._scheduled.cache_clear()

    def tearDown(self) -> None:
        """Restores the environment, clears the in-process cache, and removes the temporary directory."""
        if self.previous_cache is None:
            os.environ.pop("MINITPU_SCHEDULE_CACHE", None)
        else:
            os.environ["MINITPU_SCHEDULE_CACHE"] = self.previous_cache
        asm._scheduled.cache_clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_remembered_schedule_is_the_one_that_was_derived(self) -> None:
        """Verifies that a memo hit reproduces the image the first assembly derived."""
        kernel = _kernel()
        derived = kernel.assemble(pad_to=4)
        asm._scheduled.cache_clear()
        self.assertEqual(len(list(self.directory.glob("*.sched"))), 1)
        self.assertEqual(kernel.assemble(pad_to=4), derived)

    def test_a_corrupt_payload_is_a_miss_rather_than_an_image(self) -> None:
        """Verifies that a memo whose payload was altered is discarded and re-derived."""
        kernel = _kernel()
        derived = kernel.assemble(pad_to=4)
        remembered = next(self.directory.glob("*.sched"))
        blob = bytearray(remembered.read_bytes())
        # Flips the last byte, which is payload rather than magic or digest.
        blob[-1] ^= 0xFF
        remembered.write_bytes(bytes(blob))
        asm._scheduled.cache_clear()
        self.assertIsNone(asm._remembered_schedule(remembered.stem))
        self.assertEqual(kernel.assemble(pad_to=4), derived)

    def test_a_scheduler_change_renames_every_remembered_schedule(self) -> None:
        """Verifies that the memo key tracks the scheduler's own digest."""
        kernel = _kernel()
        before = asm._schedule_digest(kernel.bundles, "place")
        real_digest = asm._scheduler_digest
        try:
            asm._scheduler_digest = lambda: b"a different scheduler".ljust(32, b"\0")
            after = asm._schedule_digest(kernel.bundles, "place")
        finally:
            asm._scheduler_digest = real_digest
        self.assertNotEqual(before, after)
        self.assertEqual(asm._schedule_digest(kernel.bundles, "place"), before)

    def test_every_placement_is_remembered_apart(self) -> None:
        """Verifies that each placement strategy produces a distinct memo key."""
        kernel = _kernel()
        digests = {asm._schedule_digest(kernel.bundles, name) for name in asm._PLACEMENTS}
        self.assertEqual(len(digests), len(asm._PLACEMENTS))

    def test_turning_the_memo_off_still_assembles(self) -> None:
        """Verifies that a disabled memo writes nothing and still yields an image."""
        os.environ["MINITPU_SCHEDULE_CACHE"] = "off"
        asm._scheduled.cache_clear()
        kernel = _kernel()
        derived = kernel.assemble(pad_to=4)
        self.assertEqual(list(self.directory.glob("*.sched")), [])
        self.assertTrue(derived)


def _e_fields(word: int) -> tuple[int, int, int, int]:
    """Returns the opcode, destination, source A, and source B fields of a bundle's E slot."""
    slot = word >> 108
    return ((slot >> 15) & 0xF, (slot >> 10) & 0x1F, (slot >> 5) & 0x1F, slot & 0x1F)


def _image(body: Callable[[AsmBuilder], None]) -> tuple[int, ...]:
    """Returns the scheduled image of the bundle stream the given body emits."""
    builder = AsmBuilder()
    body(builder)
    return tuple(schedule(list(builder.bundles)))


def _fifo_capacity() -> int:
    """Returns the number of vmatpush operations the MXU output FIFO holds without draining."""
    depth, entries_per_push, _ = mxu_output_fifo()
    return depth // entries_per_push


def _kernel() -> asm.AsmKernel:
    """Returns a three-bundle kernel used to exercise the schedule memo."""
    builder = AsmBuilder()
    builder.bundle(x=builder.vld(vreg=0, row=0))
    builder.bundle(e=builder.vadd(vd=1, va=0, vb=0))
    builder.bundle(f=builder.halt())
    return asm.AsmKernel(name="remembered", bundles=tuple(builder.bundles))


if __name__ == "__main__":
    unittest.main()
