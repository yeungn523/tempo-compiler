"""The host half of D6: a process that dies mid-launch must not leave silently.

The device has no watchdog. A host killed between the doorbell and the retire
poll leaves `instr_ready` at 0, and the next process to open the board times out
on its own first poll and blames itself -- the only recovery being a reprogram.
These check the window is tracked, that the drain closes it, and that a device
which will not retire is REPORTED rather than left for the next user to find.
"""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stderr

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from minitpu.abi import AXI_REG, CMD_WINDOW_WORDS  # noqa: E402
from runtime import memcpy as M  # noqa: E402
from runtime.backend import MockMmio  # noqa: E402


class StuckMmio(MockMmio):
    """Idle enough to accept a doorbell, then never idle again.

    Phase 1 must succeed or the command is never issued and there is no window to
    test. So this reports ready until the doorbell is rung, and never after --
    which is what a kernel that will not retire looks like from the host.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rung = False

    def write32(self, offset: int, value: int) -> None:
        if offset == AXI_REG["cmd_w0"]:
            self.rung = True
        super().write32(offset, value)

    def read32(self, offset: int) -> int:
        if offset == AXI_REG["instr_ready"]:
            return 0 if self.rung else 1
        return super().read32(offset)


class NeverIdleMmio(MockMmio):
    """Never idle at all -- what the NEXT process sees on a wedged board."""

    def read32(self, offset: int) -> int:
        if offset == AXI_REG["instr_ready"]:
            return 0
        return super().read32(offset)


class DrainTest(unittest.TestCase):
    def setUp(self) -> None:
        M._IN_FLIGHT = None

    def tearDown(self) -> None:
        M._IN_FLIGHT = None

    def test_nothing_outstanding_drains_immediately(self) -> None:
        self.assertTrue(M.minitpuDrain())

    def test_a_completed_command_leaves_nothing_in_flight(self) -> None:
        mmio = MockMmio()
        M._ring_command(mmio, [0] * CMD_WINDOW_WORDS)
        self.assertIsNone(M._IN_FLIGHT)

    def test_an_interrupt_in_phase_three_leaves_the_window_open(self) -> None:
        """The whole point: the marker must SURVIVE the exception, not be cleared."""
        mmio = StuckMmio()
        real, M._POLL_TIMEOUT = M._POLL_TIMEOUT, 0.05
        try:
            with self.assertRaises(TimeoutError):
                M._ring_command(mmio, [0] * CMD_WINDOW_WORDS)  # phase 3 never retires
        finally:
            M._POLL_TIMEOUT = real
        self.assertIs(M._IN_FLIGHT, mmio)

    def test_the_drain_reports_a_device_that_will_not_retire(self) -> None:
        M._IN_FLIGHT = NeverIdleMmio()
        self.assertFalse(M.minitpuDrain(timeout=0.01))
        self.assertIsNone(M._IN_FLIGHT, "a failed drain must not leave the marker set")

    def test_a_recovered_device_drains_clean(self) -> None:
        """`instr_ready` returns after a few polls, as a slow kernel would."""
        M._IN_FLIGHT = MockMmio(ready_after=2)
        self.assertTrue(M.minitpuDrain(timeout=1.0))

    def test_exiting_mid_launch_says_what_to_do(self) -> None:
        M._IN_FLIGHT = NeverIdleMmio()
        M._DRAIN_TIMEOUT, real = 0.01, M._DRAIN_TIMEOUT
        try:
            captured = io.StringIO()
            with redirect_stderr(captured):
                M._drain_at_exit()
        finally:
            M._DRAIN_TIMEOUT = real
        message = captured.getvalue()
        self.assertIn("still executing on the device", message)
        self.assertIn("sudo make -C board_package minitpu", message)

    def test_exiting_cleanly_says_nothing(self) -> None:
        captured = io.StringIO()
        with redirect_stderr(captured):
            M._drain_at_exit()
        self.assertEqual(captured.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
