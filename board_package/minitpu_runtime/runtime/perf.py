"""Provides the device-side performance counters the MINITPU exposes as read-only CSRs.

They are the only measurement taken on the device itself, so they are what compares against simulator counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from minitpu.abi import AXI_REG

from .memcpy import _backend

_DEVICE_MEMORY_WORD_BYTES: int = 32
"""Byte width of one device-memory word, which is the unit the beat counter counts."""


@dataclass(frozen=True)
class PerfSample:
    """Stores one read of the three device performance counters."""

    cycles: int
    """Cycles the last launch took, from perf_cnt_cycles (RO @0x40)."""
    bundles: int
    """Bundles the last launch issued, from perf_cnt_instrs (RO @0x44)."""
    beats: int
    """Device-memory words moved since rst_n, from beat_count (RO @0x50)."""

    def __sub__(self, earlier: PerfSample) -> PerfDelta:
        """Combines this sample with a sample taken before the launch.

        Only the beat count subtracts. The tile reset zeroes the cycle and bundle counters at every launch, so
        both already hold that launch's own totals and two identical launches read identically.
        """
        return PerfDelta(
            cycles=self.cycles,
            bundles=self.bundles,
            beats=self.beats - earlier.beats,
        )


@dataclass(frozen=True)
class PerfDelta:
    """Stores the work the device did between two performance-counter samples."""

    cycles: int
    """Cycles the launch took."""
    bundles: int
    """Bundles the launch issued."""
    beats: int
    """Device-memory words moved over the launch."""

    @property
    def issue_rate(self) -> float:
        """Returns the bundles issued per cycle, where 1.0 means nothing spaced the schedule."""
        return self.bundles / self.cycles if self.cycles else 0.0

    def ddr_bytes_per_second(self, clock_hz: float) -> float:
        """Computes the DDR bandwidth the launch achieved, in bytes per second, or zero when it took no cycles.

        Args:
            clock_hz: The fabric clock frequency the counters were accumulated at.
        """
        if not self.cycles:
            return 0.0
        return self.beats * _DEVICE_MEMORY_WORD_BYTES * clock_hz / self.cycles

    def report(self, *, clock_hz: float = 200e6, blocks: int | None = None) -> str:
        """Formats the counters as one comma-separated line of cycles, bundles, bytes moved and bandwidth.

        Args:
            clock_hz: The fabric clock frequency the counters were accumulated at.
            blocks: The number of blocks the launch processed, which adds a per-block cycle cost when given.
        """
        parts = [
            f"{self.cycles} cycles",
            f"{self.bundles} bundles ({self.issue_rate:.2f}/cycle)",
            f"{self.beats * _DEVICE_MEMORY_WORD_BYTES / 1e6:.2f} MB moved",
            f"{self.ddr_bytes_per_second(clock_hz) / 1e6:.0f} MB/s DDR",
        ]
        if blocks:
            parts.insert(1, f"{self.cycles / blocks:.0f} cycles/block")
        return ", ".join(parts)


def read_perf() -> PerfSample:
    """Samples the device performance counters over the register backend."""
    mmio = _backend()
    return PerfSample(
        cycles=mmio.read32(AXI_REG["perf_cnt_cycles"]),
        bundles=mmio.read32(AXI_REG["perf_cnt_instrs"]),
        beats=mmio.read32(AXI_REG["beat_count"]),
    )
