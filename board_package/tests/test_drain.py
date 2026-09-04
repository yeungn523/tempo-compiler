"""Contains tests for the host-side drain that closes the launch window on process exit."""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stderr

# Extends the import path before the runtime imports below resolve, as the package ships its modules beside the tests
# rather than installing them.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from minitpu.abi import AXI_REG, CMD_WINDOW_WORDS  # noqa: E402
from runtime import memcpy  # noqa: E402
from runtime.backend import MockMmio  # noqa: E402


class StuckMmio(MockMmio):
    """Reports ready until the doorbell is rung and never after, as a kernel that never finishes does.

    Attributes:
        rung: Determines whether the command doorbell has been written.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rung = False

    def write32(self, offset: int, value: int) -> None:
        """Records the doorbell write and forwards the write to the mock register file."""
        if offset == AXI_REG["cmd_w0"]:
            self.rung = True
        super().write32(offset, value)

    def read32(self, offset: int) -> int:
        """Returns 0 for the ready register once the doorbell has been rung."""
        if offset == AXI_REG["instr_ready"]:
            return 0 if self.rung else 1
        return super().read32(offset)


class NeverIdleMmio(MockMmio):
    """Reports busy on every ready poll, as a wedged board does to the next process that opens it."""

    def read32(self, offset: int) -> int:
        """Returns 0 for the ready register and the mock value for every other register."""
        if offset == AXI_REG["instr_ready"]:
            return 0
        return super().read32(offset)


class DrainTest(unittest.TestCase):
    """Verifies the in-flight marker, the drain, and the exit handler that reports a stuck device."""

    def setUp(self) -> None:
        """Clears the in-flight marker before each test."""
        memcpy._IN_FLIGHT = None

    def tearDown(self) -> None:
        """Clears the in-flight marker after each test."""
        memcpy._IN_FLIGHT = None

    def test_nothing_outstanding_drains_immediately(self) -> None:
        """Verifies that a drain with no command in flight succeeds without polling."""
        self.assertTrue(memcpy.minitpuDrain())

    def test_a_completed_command_leaves_nothing_in_flight(self) -> None:
        """Verifies that a command which completes clears the in-flight marker."""
        mmio = MockMmio()
        memcpy._ring_command(mmio=mmio, words=[0] * CMD_WINDOW_WORDS)
        self.assertIsNone(memcpy._IN_FLIGHT)

    def test_an_interrupt_in_phase_three_leaves_the_window_open(self) -> None:
        """Verifies that the in-flight marker survives the timeout raised while awaiting the result."""
        mmio = StuckMmio()
        previous_timeout, memcpy._POLL_TIMEOUT = memcpy._POLL_TIMEOUT, 0.05
        try:
            with self.assertRaises(TimeoutError):
                memcpy._ring_command(mmio=mmio, words=[0] * CMD_WINDOW_WORDS)
        finally:
            memcpy._POLL_TIMEOUT = previous_timeout
        self.assertIs(memcpy._IN_FLIGHT, mmio)

    def test_the_drain_reports_a_device_that_never_finishes(self) -> None:
        """Verifies that a drain against a device that never finishes fails and still clears the marker."""
        memcpy._IN_FLIGHT = NeverIdleMmio()
        self.assertFalse(memcpy.minitpuDrain(timeout=0.01))
        self.assertIsNone(memcpy._IN_FLIGHT, "a failed drain must not leave the marker set")

    def test_a_recovered_device_drains_clean(self) -> None:
        """Verifies that a device reporting ready after a few polls drains successfully."""
        memcpy._IN_FLIGHT = MockMmio(ready_after=2)
        self.assertTrue(memcpy.minitpuDrain(timeout=1.0))

    def test_exiting_mid_launch_says_what_to_do(self) -> None:
        """Verifies that the exit handler names the recovery command when the drain fails."""
        memcpy._IN_FLIGHT = NeverIdleMmio()
        previous_timeout, memcpy._DRAIN_TIMEOUT = memcpy._DRAIN_TIMEOUT, 0.01
        try:
            captured = io.StringIO()
            with redirect_stderr(captured):
                memcpy._drain_at_exit()
        finally:
            memcpy._DRAIN_TIMEOUT = previous_timeout
        message = captured.getvalue()
        self.assertIn("still executing on the device", message)
        self.assertIn("sudo make -C board_package minitpu", message)

    def test_exiting_cleanly_says_nothing(self) -> None:
        """Verifies that the exit handler stays silent when nothing is in flight."""
        captured = io.StringIO()
        with redirect_stderr(captured):
            memcpy._drain_at_exit()
        self.assertEqual(captured.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
