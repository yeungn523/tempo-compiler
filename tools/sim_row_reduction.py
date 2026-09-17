#!/usr/bin/env python3
"""Provides the RTL check that a score tile's row statistics reduce to what a host computes."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The repository's tools directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "board_package"))
sys.path.insert(0, str(_HERE.parent / "board_package" / "minitpu_runtime"))

from bf16 import _NUM_LANES, _TILE, as_bf16_words, from_bf16_words  # noqa: E402
from kernels import build_row_statistics  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the exponential's table and the BF16 accumulate chain are allowed between them."""

_SPREAD: float = 4.0
"""Bound the drawn scores fall inside, which keeps every exponent the table covers inside its range too."""

_SEED: int = 0
"""Seed the scores are drawn under, so a failure is reproducible."""


def simulate(*, turnaround: int | None = None, seed: int = _SEED, build_dir: str | None = None,
             max_cycles: int | None = None) -> tuple[int, float, float]:
    """Runs one score tile's row statistics against the RTL and returns what each of the three cost.

    Args:
        turnaround: Bundles an access waits after the one whose VMEM rows it reads, or None for the kernel's own.
        seed: Seed the scores are drawn under.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles the round may take, or None for the bound the testbench carries.

    Returns:
        The rows whose maximum the device missed, the probabilities' relative error, and the sums'.

    Raises:
        RuntimeError: If the testbench does not report the round halted.
    """
    kernel = build_row_statistics() if turnaround is None else build_row_statistics(turnaround=turnaround)
    scores = _scores(seed=seed)
    tiles = 3
    drain_bytes = tiles * _TILE * _NUM_LANES * 2

    dump = _run_image(image=kernel.image.to_bytes(pad_to=4),
                      activations=as_bf16_words(scores).tobytes(), weights=bytes(_NUM_LANES * 2),
                      drain_bytes=drain_bytes, build_dir=build_dir, max_cycles=max_cycles,
                      what=f"the row statistics of a {_TILE}x{_NUM_LANES} score tile")
    drained = from_bf16_words(np.frombuffer(dump, dtype="<u2")[: tiles * _TILE * _NUM_LANES])
    maxima, probabilities, sums = drained.reshape(tiles, _TILE, _NUM_LANES)

    # The maximum of a set of BF16 values is one of them, so the device owes this exactly in every lane the
    # broadcast covers, which checks the reduction and the layout it leaves at once.
    expected_maxima = scores.max(axis=1)
    missed = int(np.count_nonzero((maxima != expected_maxima[:, None]).any(axis=1)))

    expected_probabilities = np.exp((scores - expected_maxima[:, None]).astype(np.float64))
    expected_sums = expected_probabilities.sum(axis=1)
    return (missed,
            float(np.abs(probabilities - expected_probabilities).max() / expected_probabilities.max()),
            float(np.abs(sums - expected_sums[:, None]).max() / expected_sums.max()))


def main() -> int:
    """Runs the row statistics against the RTL and reports what each of the three cost.

    Returns:
        Zero when every row's maximum was exact and both reduced quantities held to the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turnaround", type=int, default=None,
                        help="bundles an access waits after the one whose VMEM rows it reads")
    parser.add_argument("--seed", type=int, default=_SEED, help="seed the scores are drawn under")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles the round may take, past the bound the testbench carries")
    arguments = parser.parse_args()

    missed, probabilities, sums = simulate(turnaround=arguments.turnaround, seed=arguments.seed,
                                           build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
    verdict = "PASS" if not missed and max(probabilities, sums) <= _TOLERANCE else "FAIL"
    print(f"row statistics  {verdict}  maxima {_TILE - missed}/{_TILE} rows exact, "
          f"probabilities {probabilities * 100:.4f}%, sums {sums * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


def _scores(*, seed: int = _SEED) -> NDArray[np.float32]:
    """Returns the score tile the check runs on, at the precision the device holds it.

    Rounding here is what lets the maxima be compared exactly, the device reducing the values it was given.

    Args:
        seed: Seed the scores are drawn under.

    Returns:
        The [16, 16] tile, on the BF16 grid.
    """
    drawn = np.random.default_rng(seed).uniform(-_SPREAD, _SPREAD, size=(_TILE, _NUM_LANES)).astype(np.float32)
    return from_bf16_words(as_bf16_words(drawn)).reshape(_TILE, _NUM_LANES)


if __name__ == "__main__":
    raise SystemExit(main())
