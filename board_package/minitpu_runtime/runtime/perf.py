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

    def __sub__(self, earlier: PerfSample) -> PerfDelta:
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

    def ddr_bytes_per_second(self, f_clk_hz: float, *, launches: int = 1) -> float:
        """Achieved DDR bandwidth, the figure that ranks a design.

        **The two counters do not span the same window.** `cycles` is the LAST
        launch's, while `beats` is cumulative, so a delta covering N launches has
        to say so or this over-reports by about N. The sanity bound is the AXI
        port: 256 bits a PL cycle is 6.4 GB/s at 200 MHz, and no schedule beats
        its own port -- an answer above that means `launches` is wrong.

        Args:
            f_clk_hz: PL clock the cycle counter ran at.
            launches: Launches the beat delta covers.
        """
        if not self.cycles or launches < 1:
            return 0.0
        return self.beats * DM_WORD_BYTES * f_clk_hz / (self.cycles * launches)

    def report(self, *, f_clk_hz: float = 200e6, blocks: int | None = None,
               launches: int = 1) -> str:
        parts = [
            f"{self.cycles} cycles",
            f"{self.instrs} bundles ({self.issue_rate:.2f}/cycle)",
            f"{self.beats * DM_WORD_BYTES / 1e6:.2f} MB moved",
            f"{self.ddr_bytes_per_second(f_clk_hz, launches=launches) / 1e6:.0f} MB/s DDR",
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
