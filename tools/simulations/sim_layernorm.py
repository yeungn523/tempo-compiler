#!/usr/bin/env python3
"""Provides the RTL check that the LayerNorm image the MLIR pipeline compiles from GPT-2's kernel normalises a token
block as the host reference does.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The directory holding the RTL checks."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "board_package"))

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _ELEMENT_BYTES,
    _TILE,
    _VREG_ELEMENTS,
    _on_bf16_grid,
    as_bf16_words,
    from_bf16_words,
    pack_activations,
)
from driver import compile_kernel  # noqa: E402
from gpt2_triton import gpt2_layernorm  # noqa: E402
from references import layernorm_reference  # noqa: E402
from rtl import run_image  # noqa: E402

_TOLERANCE: float = 0.01
"""Relative error the image is allowed against the host reference, which the rsqrt table's interpolation costs."""

_MODEL: int = 768
"""GPT-2 small's model width, being the row each token normalises."""

_MAX_CYCLES: int = 50_000_000
"""Cycles one block is given, the bound the testbench carries covering a far smaller run."""


def simulate(*, build_dir: str | None = None, max_cycles: int = _MAX_CYCLES) -> float:
    """Runs one token block through the compiled LayerNorm against the RTL and returns its error.

    Args:
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take before the testbench reports it timed out.

    Returns:
        The image's error against the host reference, relative to the reference's norm.
    """
    values, gamma, beta = _operands()
    compiled = compile_kernel(gpt2_layernorm, N=_MODEL)

    # Packs the constants block as the board runtime does from the manifest entry: gamma, beta, then each scalar
    # repeated across one register.
    broadcast = [np.full(_VREG_ELEMENTS, scalar, dtype=np.float32) for scalar in compiled.entry["scalars"]]
    constants = as_bf16_words(values=np.concatenate([gamma, beta, *broadcast]))
    rows = pack_activations(activations=values)
    drain_bytes = _BLOCK_TOKENS * _MODEL * _ELEMENT_BYTES
    dumped = run_image(compiled.image, arguments=[rows.tobytes(), constants.tobytes(), bytes(drain_bytes)],
                       dump_bytes=drain_bytes, what="GPT-2 small's LayerNorm", build_dir=build_dir,
                       max_cycles=max_cycles)
    tiles = from_bf16_words(words=np.frombuffer(dumped, dtype="<u2"))
    normalised = tiles.reshape(_MODEL // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, _MODEL)
    expected = layernorm_reference(rows=values, gamma=gamma, beta=beta)
    return float(np.linalg.norm(normalised - expected) / np.linalg.norm(expected))


def _operands() -> tuple[NDArray[np.float32], ...]:
    """Returns a token block off centre, and a gamma and beta near the identity, all on the BF16 grid."""
    rng = np.random.default_rng(0)
    return (_on_bf16_grid(values=(rng.standard_normal((_BLOCK_TOKENS, _MODEL)) * 2.0 + 0.5).astype(np.float32)),
            _on_bf16_grid(values=(1.0 + 0.1 * rng.standard_normal(_MODEL)).astype(np.float32)),
            _on_bf16_grid(values=(0.1 * rng.standard_normal(_MODEL)).astype(np.float32)))


def main() -> int:
    """Runs the LayerNorm image against the RTL and reports its error.

    Returns:
        Zero when the image matches the reference within the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=_MAX_CYCLES,
                        help="cycles one round may take before the testbench reports it timed out")
    arguments = parser.parse_args()

    error = simulate(build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
    verdict = "PASS" if error <= _TOLERANCE else "FAIL"
    print(f"{_BLOCK_TOKENS}x{_MODEL} LayerNorm  {verdict}  relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
