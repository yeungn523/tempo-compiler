#!/usr/bin/env python3
"""Provides the RTL check that the compiled LayerNorm image computes what the hand-written builder does.
"""

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

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _ELEMENT_BYTES,
    _TILE,
    _VREG_ELEMENTS,
    _on_bf16_grid,
    as_bf16_words,
    from_bf16_words,
    layernorm_constants,
    pack_activations,
)
from build_gpt2_images import _compile_rows  # noqa: E402
from tests.hand_images import layernorm_rows  # noqa: E402
from references import layernorm_reference  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.01
"""Relative error the image is allowed against the host reference, which the rsqrt table's interpolation costs."""

_MODEL: int = 768
"""GPT-2 small's model width, being the row each token normalises."""

_MAX_CYCLES: int = 50_000_000
"""Cycles one block is given, the bound the testbench carries covering a far smaller run."""


def simulate(*, build_dir: str | None = None, max_cycles: int = _MAX_CYCLES) -> tuple[float, bool]:
    """Runs one token block through the compiled and the hand-written LayerNorm against the RTL.

    Args:
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take before the testbench reports it timed out.

    Returns:
        The compiled image's error against the host reference, and whether its output equals the hand builder's
        bit for bit.
    """
    values, gamma, beta = _operands()
    image, entry = _compile_rows(kind="layernorm", features=_MODEL)
    broadcast = [np.full(_VREG_ELEMENTS, scalar, dtype=np.float32) for scalar in entry["scalars"]]
    compiled = _normalised(image=image, values=values, build_dir=build_dir, max_cycles=max_cycles,
                           constants=as_bf16_words(values=np.concatenate([gamma, beta, *broadcast])))
    hand = _normalised(image=layernorm_rows(),
                       values=values, constants=layernorm_constants(gamma=gamma, beta=beta), build_dir=build_dir,
                       max_cycles=max_cycles)
    expected = layernorm_reference(rows=values, gamma=gamma, beta=beta)
    return float(np.linalg.norm(compiled - expected) / np.linalg.norm(expected)), bool(np.array_equal(compiled, hand))


def _normalised(*, image: bytes, values: NDArray[np.float32], constants: NDArray[np.uint16], build_dir: str | None,
                max_cycles: int) -> NDArray[np.float32]:
    """Runs one LayerNorm image over a token block and returns the rows it writes, in row-major order."""
    drain_bytes = _BLOCK_TOKENS * _MODEL * _ELEMENT_BYTES
    dumped = _run_image(image=image, activations=pack_activations(activations=values).tobytes(),
                        weights=constants.tobytes(), drain_bytes=drain_bytes, build_dir=build_dir,
                        max_cycles=max_cycles, what="GPT-2 small's LayerNorm")
    rows = from_bf16_words(words=np.frombuffer(dumped, dtype="<u2")[: drain_bytes // _ELEMENT_BYTES])
    return rows.reshape(_MODEL // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, _MODEL)


def _operands() -> tuple[NDArray[np.float32], ...]:
    """Returns a token block off centre, and a gamma and beta near the identity, all on the BF16 grid."""
    rng = np.random.default_rng(0)
    return (_on_bf16_grid(values=(rng.standard_normal((_BLOCK_TOKENS, _MODEL)) * 2.0 + 0.5).astype(np.float32)),
            _on_bf16_grid(values=(1.0 + 0.1 * rng.standard_normal(_MODEL)).astype(np.float32)),
            _on_bf16_grid(values=(0.1 * rng.standard_normal(_MODEL)).astype(np.float32)))


def main() -> int:
    """Runs both images against the RTL and reports the compiled one's error.

    Returns:
        Zero when the compiled image matches the hand builder bit for bit and the reference within the tolerance.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=_MAX_CYCLES,
                        help="cycles one round may take before the testbench reports it timed out")
    arguments = parser.parse_args()

    error, identical = simulate(build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
    verdict = "PASS" if identical and error <= _TOLERANCE else "FAIL"
    print(f"{_BLOCK_TOKENS}x{_MODEL} LayerNorm  {verdict}  relative error {error * 100:.4f}%, "
          f"{'bit identical to' if identical else 'differs from'} the hand-written builder")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
