"""Holds the LayerNorm the MLIR pipeline builds to the host reference and to the image codegen/ compiles from the same
Triton kernel, through the emulator."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from bf16 import _BLOCK_TOKENS, _TILE, _VREG_ELEMENTS, as_bf16_words, from_bf16_words, pack_activations  # noqa: E402
from build_gpt2_images import _compile_rows  # noqa: E402
from build_mlir_images import _TOOLS, compile_kernel, layernorm_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image, to_bf16  # noqa: E402
from references import layernorm_reference  # noqa: E402

pytestmark = pytest.mark.skipif(not (_TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")

_FEATURES: int = 768
"""Stores the width GPT-2 small normalises over, the one width the host reference covers."""


def _run_layernorm(image: pathlib.Path, rows: np.ndarray, constants: np.ndarray) -> tuple[np.ndarray, int]:
    """Runs one LayerNorm image over a block and returns the rows it drains and the cycles it took."""
    features = rows.shape[1]
    words = pack_activations(activations=rows)
    rows_address = 4096
    constants_address = rows_address + words.nbytes + 4096
    output_address = constants_address + constants.nbytes + 4096
    machine = Machine(np.zeros(output_address + words.nbytes + 4096, dtype=np.uint8),
                      arguments=(rows_address, constants_address, output_address))
    machine.write_words(rows_address, words)
    machine.write_words(constants_address, constants)
    machine.run(read_image(image))
    tiles = from_bf16_words(words=machine.read_words(output_address, words.size))
    normalised = tiles.reshape(features // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2)
    return normalised.reshape(_BLOCK_TOKENS, features), machine.cycles


def test_mlir_layernorm_matches_the_reference_and_beats_codegen(tmp_path: pathlib.Path) -> None:
    """Verifies that the MLIR-built LayerNorm is closer to the host reference than codegen/'s and takes fewer cycles."""
    codegen, entry = _compile_rows(kind="layernorm", features=_FEATURES)
    baseline = tmp_path / "codegen.bin"
    baseline.write_bytes(codegen)
    image = tmp_path / "layernorm.bin"
    compile_kernel(layernorm_kernel(features=_FEATURES), image=image)

    # Draws the board check's operands, and packs the constants as the host runtime does from the manifest entry.
    rng = np.random.default_rng(0)
    rows = to_bf16((rng.standard_normal((_BLOCK_TOKENS, _FEATURES)) * 2.0).astype(np.float32))
    gamma = to_bf16((1.0 + 0.1 * rng.standard_normal(_FEATURES)).astype(np.float32))
    beta = to_bf16((0.1 * rng.standard_normal(_FEATURES)).astype(np.float32))
    broadcast = [np.full(_VREG_ELEMENTS, scalar, dtype=np.float32) for scalar in entry["scalars"]]
    constants = as_bf16_words(values=np.concatenate([gamma, beta, *broadcast]))
    expected = layernorm_reference(rows=rows, gamma=gamma, beta=beta)

    normalised, cycles = _run_layernorm(image, rows, constants)
    baseline_normalised, baseline_cycles = _run_layernorm(baseline, rows, constants)
    error = np.linalg.norm(normalised - expected) / np.linalg.norm(expected)
    baseline_error = np.linalg.norm(baseline_normalised - expected) / np.linalg.norm(expected)
    assert error <= baseline_error
    assert cycles < baseline_cycles


@pytest.mark.parametrize("features", [128, 512, 1024, 2048])
def test_mlir_layernorm_covers_every_width_it_accepts(features: int, tmp_path: pathlib.Path) -> None:
    """Verifies that LayerNorms narrower and wider than GPT-2's, the wider ones streaming each token from its buffer,
    stay within the board's bound of a float64 reference."""
    image = tmp_path / "layernorm.bin"
    compile_kernel(layernorm_kernel(features=features), image=image)

    rng = np.random.default_rng(features)
    rows = to_bf16((rng.standard_normal((_BLOCK_TOKENS, features)) * 2.0 + 0.5).astype(np.float32))
    gamma = to_bf16((1.0 + 0.1 * rng.standard_normal(features)).astype(np.float32))
    beta = to_bf16((0.1 * rng.standard_normal(features)).astype(np.float32))
    scalars = (1.0 / features, 9.99999974e-06)
    broadcast = [np.full(_VREG_ELEMENTS, scalar, dtype=np.float32) for scalar in scalars]
    constants = as_bf16_words(values=np.concatenate([gamma, beta, *broadcast]))

    values = rows.astype(np.float64)
    centred = values - values.mean(axis=1, keepdims=True)
    expected = centred / np.sqrt((centred**2).mean(axis=1, keepdims=True) + scalars[1]) * gamma + beta
    normalised, _ = _run_layernorm(image, rows, constants)
    assert np.linalg.norm(normalised - expected) / np.linalg.norm(expected) < 0.01
