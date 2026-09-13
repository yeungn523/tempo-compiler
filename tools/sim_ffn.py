#!/usr/bin/env python3
"""Provides the RTL check that the compiled fused image computes GPT-2 small's feed-forward layer.
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
sys.path.insert(0, str(_HERE.parent / "board_package" / "minitpu_runtime"))

from asm import AsmKernel  # noqa: E402
from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _ELEMENT_BYTES,
    _TILE,
    _ZERO_TILE_WORDS,
    from_bf16_words,
    fused_ffn_bf16_reference,
    fused_ffn_reference,
    pack_fused_ffn,
)
from emit import fused_ffn_stream  # noqa: E402
from sim_gemm import _run_image  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the layer is allowed past what the BF16 format costs over its own contraction."""

_WEIGHT_SCALE: float = 0.25
"""Scale the drawn weights carry, which keeps the intermediate inside the range the SFU's table covers."""

_MODEL: int = 768
"""GPT-2 small's model width, being the contraction the up projection walks and the width the layer drains."""

_HIDDEN: int = 3072
"""GPT-2 small's feed-forward width, being what the up projection expands its input to."""

_MAX_CYCLES: int = 200_000_000
"""Cycles the layer is given, the bound the testbench carries covering a far smaller run."""


def simulate(*, build_dir: str | None = None, max_cycles: int = _MAX_CYCLES) -> tuple[float, float]:
    """Runs GPT-2 small's feed-forward layer against the RTL and returns what it and a BF16 model each cost.

    Args:
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take before the testbench reports it timed out.

    Returns:
        The layer's error against the exact reference, and the error a model rounding where the machine rounds
        carries against that same reference. Over a contraction this deep the second is the bar worth reading,
        the first counting the format's own loss alongside anything the image got wrong.
    """
    steps, chunks, output_tiles = _MODEL // _TILE, _HIDDEN // _TILE, _MODEL // _TILE
    activations, up_weight, down_weight = _operands()
    staged, weights = pack_fused_ffn(activations, up_weight, down_weight)
    bundles, _ = fused_ffn_stream(chunks=chunks, steps=steps, output_tiles=output_tiles)
    image = AsmKernel(name="fused_ffn", bundles=bundles).to_bytes(pad_to=4)

    drain_words = _ZERO_TILE_WORDS + _BLOCK_TOKENS * output_tiles * _TILE
    dumped = _run_image(image=image,
                        activations=staged.tobytes(), weights=weights.tobytes(),
                        drain_bytes=drain_words * _ELEMENT_BYTES, build_dir=build_dir,
                        max_cycles=max_cycles, what="GPT-2 small's feed-forward layer")

    words = np.frombuffer(dumped, dtype="<u2")
    # The sweep drains tile by tile, each one a token block, which the layer's rows read back from.
    drained = from_bf16_words(words=words[_ZERO_TILE_WORDS:drain_words])
    product = drained.reshape(output_tiles, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(
        _BLOCK_TOKENS, _MODEL)

    expected = fused_ffn_reference(activations, up_weight, down_weight)
    scale = max(float(np.abs(expected).max()), 1e-9)
    modelled = fused_ffn_bf16_reference(activations, up_weight, down_weight)
    return (float(np.abs(product - expected).max() / scale),
            float(np.abs(modelled - expected).max() / scale))


def _operands() -> tuple[NDArray[np.float32], ...]:
    """Returns an input and both projections, drawn at the layer's extents."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((_BLOCK_TOKENS, _MODEL), dtype=np.float32),
            (rng.standard_normal((_MODEL, _HIDDEN), dtype=np.float32) * _WEIGHT_SCALE).astype(np.float32),
            (rng.standard_normal((_HIDDEN, _MODEL), dtype=np.float32) * _WEIGHT_SCALE).astype(np.float32))


def main() -> int:
    """Runs the layer against the RTL and reports its error.

    Returns:
        Zero when the layer matched within the tolerance the BF16 accumulate chain allows.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=_MAX_CYCLES,
                        help="cycles one round may take before the testbench reports it timed out")
    arguments = parser.parse_args()

    error, format_cost = simulate(build_dir=arguments.build_dir, max_cycles=arguments.max_cycles)
    what = f"{_BLOCK_TOKENS}x{_MODEL} @ {_MODEL}x{_HIDDEN} -> gelu -> {_HIDDEN}x{_MODEL}"
    verdict = "PASS" if error <= format_cost + _TOLERANCE else "FAIL"
    print(f"{what}  {verdict}  relative error {error * 100:.4f}%, "
          f"of which BF16 over this contraction costs {format_cost * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
