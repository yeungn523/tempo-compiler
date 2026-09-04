"""Device-side performance counters.

Host wall-clock around a launch measures COPY, doorbell polling and Python
overhead as well as the kernel. These counters are the only measurement taken
on the device itself, so they are what compares directly against simulator
cycle counts.

    perf_cnt_cycles  cycles the last launch took   (RO @0x40)
    perf_cnt_instrs  bundles the last launch issued (RO @0x44)
    beat_count       256-bit DM words moved         (RO @0x50)

The first two are PER-LAUNCH: the tile reset zeroes them at each launch, so
they read as that launch's own totals and two identical launches read
identically. beat_count is cumulative from rst_n, so only it is a difference.
Measured on a ZCU104: two identical GEMM launches both read cycles=723201,
instrs=317292 while beats advanced.
"""

from __future__ import annotations

from dataclasses import dataclass

from minitpu.abi import AXI_REG

from .memcpy import _backend

#: One DM word on the wire. beat_count counts 256-bit words.
DM_WORD_BYTES = 32


@dataclass(frozen=True)
class PerfSample:
    """One read of the device counters."""

    cycles: int
    instrs: int
    beats: int

    def __sub__(self, earlier: "PerfSample") -> "PerfDelta":
        """Combine an after-sample with a before-sample.

        Only beats subtracts: cycles and instrs are already this launch's own
        totals, so differencing them would report ~zero for repeated work.
        """
        return PerfDelta(
            cycles=self.cycles,
            instrs=self.instrs,
            beats=self.beats - earlier.beats,
        )


@dataclass(frozen=True)
class PerfDelta:
    """What the device did between two samples."""

    cycles: int
    instrs: int
    beats: int

    @property
    def issue_rate(self) -> float:
        """Bundles issued per cycle. 1.0 means nothing spaced the schedule."""
        return self.instrs / self.cycles if self.cycles else 0.0

    def ddr_bytes_per_second(self, f_clk_hz: float) -> float:
        """Achieved DDR bandwidth, the figure that ranks a design."""
        if not self.cycles:
            return 0.0
        return self.beats * DM_WORD_BYTES * f_clk_hz / self.cycles

    def report(self, *, f_clk_hz: float = 200e6, blocks: int | None = None) -> str:
        parts = [
            f"{self.cycles} cycles",
            f"{self.instrs} bundles ({self.issue_rate:.2f}/cycle)",
            f"{self.beats * DM_WORD_BYTES / 1e6:.2f} MB moved",
            f"{self.ddr_bytes_per_second(f_clk_hz) / 1e6:.0f} MB/s DDR",
        ]
        if blocks:
            parts.insert(1, f"{self.cycles / blocks:.0f} cycles/block")
        return ", ".join(parts)


def read_perf() -> PerfSample:
    """Sample the device counters. Call either side of a launch and subtract."""
    mmio = _backend()
    return PerfSample(
        cycles=mmio.read32(AXI_REG["perf_cnt_cycles"]),
        instrs=mmio.read32(AXI_REG["perf_cnt_instrs"]),
        beats=mmio.read32(AXI_REG["beat_count"]),
    )
