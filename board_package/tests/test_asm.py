"""Cross-check the public assembler's physical E-slot encodings."""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import asm  # noqa: E402
from asm import AsmBuilder  # noqa: E402


def e_fields(word: int) -> tuple[int, int, int, int]:
    """Return opcode, destination, source A, and source B from a bundle."""
    e = word >> 108
    return ((e >> 15) & 0xF, (e >> 10) & 0x1F, (e >> 5) & 0x1F, e & 0x1F)


class AssemblerEncodingTest(unittest.TestCase):
    def test_layernorm_e_slot_encodings(self) -> None:
        asm = AsmBuilder()
        self.assertEqual(e_fields(asm.vadd(3, 4, 5)), (1, 3, 4, 5))
        self.assertEqual(e_fields(asm.vsub(6, 7, 8)), (2, 6, 7, 8))
        self.assertEqual(e_fields(asm.vmul(9, 10, 11)), (3, 9, 10, 11))
        self.assertEqual(e_fields(asm.vrsqrt(12, 13)), (8, 12, 13, 0))
        self.assertEqual(e_fields(asm.vredsum(14, 15)), (9, 14, 15, 0))
        self.assertEqual(e_fields(asm.vredmax(16, 17)), (10, 16, 17, 0))


class SchedulingTest(unittest.TestCase):
    """The machine has no interlocks, so every emitted image must be spaced.

    These distances are the published contract in docs/isa_latency.json, which
    tb/tb_isa_conformance.sv asserts against the RTL's own WB_W_* constants.
    """

    @staticmethod
    def delay_of(word: int) -> int:
        """Reads a bundle's delay field.

        Delegates to the assembler rather than keeping a second copy of the bit
        position: this test held its own `shift = ... else 1` and would have
        silently read the wrong field the moment any slot width changed.
        """
        return asm._delay_of(word)

    def build(self, body) -> tuple[int, ...]:
        asm_kernel = AsmBuilder()
        body(asm_kernel)
        from asm import schedule  # noqa: E402
        return tuple(schedule(list(asm_kernel.bundles)))

    def test_load_to_consumer_is_spaced_to_L7(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0))       # vld W=6 -> consumer legal at +7
            a.bundle(e=a.vadd(1, 0, 0))   # reads V0
            a.bundle(f=a.halt())
        words = self.build(body)
        self.assertEqual(self.delay_of(words[0]), 6)

    def test_independent_bundles_need_no_delay(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0))
            a.bundle(x=a.vld(1, 1))       # different VREG, W=6 vs W=6 one apart
            a.bundle(f=a.halt())
        words = self.build(body)
        self.assertEqual(self.delay_of(words[0]), 0)

    def test_write_port_collision_is_separated(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0))       # lands at +6
            a.bundle(e=a.vadd(2, 3, 4))   # ALU W=5 would also land at +6
            a.bundle(f=a.halt())
        words = self.build(body)
        # The ALU must not share the write port with the load's writeback.
        self.assertGreaterEqual(self.delay_of(words[0]), 1)

    def test_vmatpush_gap_exceeds_one_delay_field_and_gets_a_filler(self) -> None:
        def body(a):
            a.bundle(m=a.vmatpush(0))     # occupies 83 cycles; one field holds 63
            a.bundle(m=a.vmatpop(1))
            a.bundle(f=a.halt())
        words = self.build(body)
        self.assertGreater(len(words), 3, "expected a filler bundle to be inserted")


class PortAExclusivityTest(unittest.TestCase):
    """VREG read port A is one physical read the matrix controller wins.

    Nothing interlocks behind that mux and no delay field can separate two
    consumers inside one bundle, so co-issuing them is rejected at assembly
    time rather than silently reading the wrong register on silicon.
    """

    def schedule_body(self, body) -> tuple[int, ...]:
        from asm import schedule  # noqa: E402

        asm_kernel = AsmBuilder()
        body(asm_kernel)
        return tuple(schedule(list(asm_kernel.bundles)))

    def test_sfu_co_issued_with_vmatpush_is_rejected(self) -> None:
        def body(a):
            a.bundle(m=a.vmatpush(12), e=a.vgelu(16, 16))
            a.bundle(f=a.halt())
        with self.assertRaisesRegex(ValueError, "read VREG port A"):
            self.schedule_body(body)

    def test_vst_co_issued_with_vmatload_is_rejected(self) -> None:
        def body(a):
            a.bundle(m=a.vmatload(24), x=a.vst(0, 0))
            a.bundle(f=a.halt())
        with self.assertRaisesRegex(ValueError, "read VREG port A"):
            self.schedule_body(body)

    def test_vld_and_vmatpop_are_not_port_a_readers(self) -> None:
        def body(a):
            a.bundle(m=a.vmatpush(12), x=a.vld(0, 0))   # vld reads VMEM
            a.bundle(m=a.vmatpop(28), e=a.vadd(1, 2, 3))  # vmatpop only writes
            a.bundle(f=a.halt())
        self.assertGreaterEqual(len(self.schedule_body(body)), 3)



class MatrixFifoTest(unittest.TestCase):
    """The MXU's per-lane output FIFO is finite and hardware never replays.

    One vmatpush streams NUM_SUBLANES activation rows and so produces that many
    results; the array's 82-cycle latency means the pops that would drain them
    cannot have run yet. A burst longer than the FIFO drops results silently, and
    no delay can fix it, because this scheduler never reorders.
    """

    def schedule_body(self, body) -> tuple[int, ...]:
        from asm import schedule  # noqa: E402

        asm_kernel = AsmBuilder()
        body(asm_kernel)
        return tuple(schedule(list(asm_kernel.bundles)))

    @staticmethod
    def pushes_that_fit() -> int:
        from asm import mxu_output_fifo  # noqa: E402

        depth, entries_per_push, _ = mxu_output_fifo()
        return depth // entries_per_push

    def test_one_push_past_capacity_is_rejected(self) -> None:
        pushes = self.pushes_that_fit() + 1

        def body(a):
            for vreg in range(pushes):
                a.bundle(m=a.vmatpush(12 + vreg))
            a.bundle(f=a.halt())
        with self.assertRaisesRegex(ValueError, "overflows the MXU output FIFO"):
            self.schedule_body(body)

    def test_capacity_pushes_fill_it_exactly(self) -> None:
        pushes = self.pushes_that_fit()

        def body(a):
            for vreg in range(pushes):
                a.bundle(m=a.vmatpush(12 + vreg))
            for vreg in range(pushes):
                a.bundle(m=a.vmatpop(vreg))
            a.bundle(f=a.halt())
        self.assertGreaterEqual(len(self.schedule_body(body)), 2 * pushes + 1)

    def test_draining_between_bursts_lets_more_pushes_through(self) -> None:
        pushes = self.pushes_that_fit()

        def body(a):
            for _ in range(2):
                for vreg in range(pushes):
                    a.bundle(m=a.vmatpush(12 + vreg))
                for vreg in range(pushes):
                    a.bundle(m=a.vmatpop(vreg))
            a.bundle(f=a.halt())
        self.assertGreaterEqual(len(self.schedule_body(body)), 4 * pushes + 1)


class FlushSlotTest(unittest.TestCase):
    """A flush waits on a done bit only a descriptor asserts.

    Flushing a slot the kernel never issued on blocks forever, with no timeout and no error, and the
    device stays wedged until the PL is reprogrammed. Slots are counted over the whole stream, because a
    loop body's flush legitimately collects a descriptor issued in the previous iteration.
    """

    @staticmethod
    def schedule_body(body) -> tuple[int, ...]:
        from asm import schedule  # noqa: E402

        asm_kernel = AsmBuilder()
        body(asm_kernel)
        return tuple(schedule(list(asm_kernel.bundles)))

    def test_flushing_a_slot_never_issued_on_is_rejected(self) -> None:
        def body(a):
            a.bundle(d=a.dma_load(0, 0, 4, 0, 3))
            a.bundle(f=a.flush_slot(3))
            a.bundle(f=a.halt())
        with self.assertRaisesRegex(ValueError, "waits on DMA slot"):
            self.schedule_body(body)

    def test_flushing_only_the_issued_slot_is_accepted(self) -> None:
        def body(a):
            a.bundle(d=a.dma_load(0, 0, 4, 0, 3))
            a.bundle(f=a.flush_slot(1))
            a.bundle(f=a.halt())
        self.assertTrue(self.schedule_body(body))

    def test_both_slots_issued_lets_the_pair_flush(self) -> None:
        def body(a):
            a.bundle(d=a.dma_load(0, 0, 4, 0, 3))
            a.bundle(d=a.dma_load(1, 8, 4, 2, 3))
            a.bundle(f=a.flush_slot(3))
            a.bundle(f=a.halt())
        self.assertTrue(self.schedule_body(body))

    def test_a_descriptor_issued_after_the_flush_still_counts(self) -> None:
        def body(a):
            a.bundle(d=a.dma_load(0, 0, 4, 0, 3))
            a.bundle(f=a.flush_slot(3))
            a.bundle(d=a.dma_load(1, 8, 4, 2, 3))
            a.bundle(f=a.halt())
        self.assertTrue(self.schedule_body(body))


class SublaneConflictTest(unittest.TestCase):
    """A strided VMEM access holds the compute port for one cycle per collision.

    bank[s] = (row + s*stride) mod NUM_SUBLANES, so adding the row rotates which
    banks are touched and never how many: the cost is a function of the stride
    and mask alone, both immediates. vpu_vmem's busy_o is observation only and
    does not gate acceptance, so issuing during it is a software error with
    nothing behind it -- which is why schedule() has to price it.
    """

    def test_odd_strides_are_conflict_free(self) -> None:
        from asm import sublane_conflict_cycles  # noqa: E402

        for stride in (1, 3, 5, 7):
            self.assertEqual(sublane_conflict_cycles(stride, 0xF), 1, f"stride {stride}")

    def test_even_strides_collide(self) -> None:
        from asm import sublane_conflict_cycles  # noqa: E402

        self.assertEqual(sublane_conflict_cycles(2, 0xF), 2)
        self.assertEqual(sublane_conflict_cycles(4, 0xF), 4)
        self.assertEqual(sublane_conflict_cycles(8, 0xF), 4)

    def test_a_mask_can_dodge_a_collision(self) -> None:
        from asm import sublane_conflict_cycles  # noqa: E402

        self.assertEqual(sublane_conflict_cycles(4, 0b0001), 1)
        self.assertEqual(sublane_conflict_cycles(4, 0b0101), 2)

    def test_a_conflicting_load_delays_the_next_access(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0, stride=4))   # 4-way conflict, holds the port 4 cycles
            a.bundle(x=a.vld(1, 64))
            a.bundle(f=a.halt())
        asm_kernel = AsmBuilder()
        body(asm_kernel)
        from asm import schedule  # noqa: E402
        words = schedule(list(asm_kernel.bundles))
        self.assertGreaterEqual(self.delay_of(words[0]), 3)

    @staticmethod
    def delay_of(word: int) -> int:
        """Reads a bundle's delay field.

        Delegates to the assembler rather than keeping a second copy of the bit
        position: this test held its own `shift = ... else 1` and would have
        silently read the wrong field the moment any slot width changed.
        """
        return asm._delay_of(word)

    def test_a_conflict_free_load_does_not(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0))
            a.bundle(x=a.vld(1, 64))
            a.bundle(f=a.halt())
        asm_kernel = AsmBuilder()
        body(asm_kernel)
        from asm import schedule  # noqa: E402
        words = schedule(list(asm_kernel.bundles))
        self.assertEqual(self.delay_of(words[0]), 0)

    def test_a_conflicting_load_also_writes_back_later(self) -> None:
        """Measured on the RTL: a 4-way conflicting vld's consumer is legal at +10, not +7."""
        def body(a):
            a.bundle(x=a.vld(0, 0, stride=4))
            a.bundle(e=a.vadd(1, 0, 0))     # consumes V0
            a.bundle(f=a.halt())
        asm_kernel = AsmBuilder()
        body(asm_kernel)
        from asm import schedule  # noqa: E402
        words = schedule(list(asm_kernel.bundles))
        self.assertEqual(self.delay_of(words[0]), 9)

    def test_a_conflict_free_load_keeps_L7(self) -> None:
        def body(a):
            a.bundle(x=a.vld(0, 0))
            a.bundle(e=a.vadd(1, 0, 0))
            a.bundle(f=a.halt())
        asm_kernel = AsmBuilder()
        body(asm_kernel)
        from asm import schedule  # noqa: E402
        words = schedule(list(asm_kernel.bundles))
        self.assertEqual(self.delay_of(words[0]), 6)

    def test_a_zero_mask_is_rejected(self) -> None:
        asm_kernel = AsmBuilder()
        with self.assertRaisesRegex(ValueError, "at least one"):
            asm_kernel.vld(0, 0, mask=0)


class RememberedScheduleTest(unittest.TestCase):
    """The cross-process memo must be a memo: a hit is identical, a doubt is a miss.

    A remembered schedule is launched without being re-derived, and this machine
    has no interlocks -- a stale or corrupt image returns plausible numbers rather
    than faulting. So each of these checks a way the memo could be wrong, not just
    that it is fast.
    """

    def setUp(self) -> None:
        self.directory = pathlib.Path(tempfile.mkdtemp())
        self.previous = os.environ.get("MINITPU_SCHEDULE_CACHE")
        os.environ["MINITPU_SCHEDULE_CACHE"] = str(self.directory)
        asm._scheduled.cache_clear()

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("MINITPU_SCHEDULE_CACHE", None)
        else:
            os.environ["MINITPU_SCHEDULE_CACHE"] = self.previous
        asm._scheduled.cache_clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    @staticmethod
    def _kernel() -> asm.AsmKernel:
        builder = AsmBuilder()
        builder.bundle(x=builder.vld(0, 0, stride=4))
        builder.bundle(e=builder.vadd(1, 0, 0))
        builder.bundle(f=builder.halt())
        return asm.AsmKernel("remembered", tuple(builder.bundles))

    def test_a_remembered_schedule_is_the_one_that_was_derived(self) -> None:
        kernel = self._kernel()
        derived = kernel.assemble(pad_to=4)
        asm._scheduled.cache_clear()
        self.assertEqual(len(list(self.directory.glob("*.sched"))), 1)
        self.assertEqual(kernel.assemble(pad_to=4), derived)

    def test_a_corrupt_payload_is_a_miss_rather_than_an_image(self) -> None:
        kernel = self._kernel()
        derived = kernel.assemble(pad_to=4)
        remembered = next(self.directory.glob("*.sched"))
        blob = bytearray(remembered.read_bytes())
        blob[-1] ^= 0xFF                      # the payload, past magic and digest
        remembered.write_bytes(bytes(blob))
        asm._scheduled.cache_clear()
        self.assertIsNone(asm._remembered_schedule(remembered.stem))
        self.assertEqual(kernel.assemble(pad_to=4), derived)

    def test_a_scheduler_change_renames_every_remembered_schedule(self) -> None:
        kernel = self._kernel()
        before = asm._schedule_digest(kernel.bundles, "place")
        real = asm._scheduler_digest
        try:
            asm._scheduler_digest = lambda: b"a different scheduler".ljust(32, b"\0")
            after = asm._schedule_digest(kernel.bundles, "place")
        finally:
            asm._scheduler_digest = real
        self.assertNotEqual(before, after)
        self.assertEqual(asm._schedule_digest(kernel.bundles, "place"), before)

    def test_every_placement_is_remembered_apart(self) -> None:
        kernel = self._kernel()
        digests = {asm._schedule_digest(kernel.bundles, name) for name in asm._PLACEMENTS}
        self.assertEqual(len(digests), len(asm._PLACEMENTS))

    def test_turning_the_memo_off_still_assembles(self) -> None:
        os.environ["MINITPU_SCHEDULE_CACHE"] = "off"
        asm._scheduled.cache_clear()
        kernel = self._kernel()
        derived = kernel.assemble(pad_to=4)
        self.assertEqual(list(self.directory.glob("*.sched")), [])
        self.assertTrue(derived)


if __name__ == "__main__":
    unittest.main()
