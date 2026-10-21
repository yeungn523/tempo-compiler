"""Holds the GEMM the MLIR pipeline builds to the board check's own operands and reference, through the emulator."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _ZERO_TILE_WORDS,
    build_operands,
    from_bf16_words,
    pack_activations,
    pack_bias,
    pack_weights,
)
from build_images import gemm_kernel  # noqa: E402
from driver import TOOLS, compile_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image, to_bf16  # noqa: E402

pytestmark = pytest.mark.skipif(not (TOOLS / "tempo-opt").exists(), reason="mlir/ is not built")


@pytest.mark.parametrize("kind", ["structured", "varying"])
def test_mlir_gemm_matches_the_board_reference(kind: str, tmp_path: pathlib.Path) -> None:
    """Verifies that the MLIR-built 768x768 GEMM is exact on structured operands and within the board's bound on varying
    ones."""
    depth = columns = 768
    image = tmp_path / "gemm.bin"
    image.write_bytes(compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=False)).image)

    left, right = build_operands(_BLOCK_TOKENS, depth, columns, kind=kind, seed=0)
    activations, weights = pack_activations(activations=left), pack_weights(weights=right)
    drain_words = _ZERO_TILE_WORDS + columns // _TILE * _BLOCK_TOKENS * _TILE
    activation_address = 4096
    weight_address = activation_address + activations.nbytes + 4096
    drain_address = weight_address + weights.nbytes + 4096
    machine = Machine(np.zeros(drain_address + 2 * drain_words + 4096, dtype=np.uint8),
                      arguments=(activation_address, weight_address, drain_address))
    machine.write_words(activation_address, activations)
    machine.write_words(weight_address, weights)
    machine.run(read_image(image))

    tiles = from_bf16_words(words=machine.read_words(drain_address, drain_words)[_ZERO_TILE_WORDS:])
    product = tiles.reshape(columns // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)
    if kind == "structured":
        np.testing.assert_array_equal(product, (left @ right).astype(np.float32))
    else:
        expected = left.astype(np.float64) @ right.astype(np.float64)
        assert np.linalg.norm(product - expected) / np.linalg.norm(expected) < 0.01


def _run_gemm(image: pathlib.Path, left: np.ndarray, right: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Runs one GEMM image over a block, binding the bias as the fourth argument when one is given, and returns the
    product it drains."""
    columns = right.shape[1]
    activations, weights = pack_activations(activations=left), pack_weights(weights=right)
    bias_words = pack_bias(bias=bias) if bias is not None else np.zeros(0, dtype=np.uint16)
    drain_words = _ZERO_TILE_WORDS + columns // _TILE * _BLOCK_TOKENS * _TILE
    activation_address = 4096
    weight_address = activation_address + activations.nbytes + 4096
    drain_address = weight_address + weights.nbytes + 4096
    bias_address = drain_address + 2 * drain_words + 4096
    arguments = (activation_address, weight_address, drain_address)
    machine = Machine(np.zeros(bias_address + bias_words.nbytes + 4096, dtype=np.uint8),
                      arguments=arguments if bias is None else (*arguments, bias_address))
    machine.write_words(activation_address, activations)
    machine.write_words(weight_address, weights)
    machine.write_words(bias_address, bias_words)
    machine.run(read_image(image))
    tiles = from_bf16_words(words=machine.read_words(drain_address, drain_words)[_ZERO_TILE_WORDS:])
    return tiles.reshape(columns // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)


@pytest.mark.parametrize(("depth", "columns", "activate"), [(768, 768, False), (3072, 768, True)])
def test_mlir_gemm_folds_the_bias(depth: int, columns: int, activate: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that a GEMM folding the bias into its accumulators drains the product plus the bias as closely as the
    plain GEMM drains the product, for a plain product and one activating its input."""
    plain, fused = tmp_path / "gemm.bin", tmp_path / "gemm_bias.bin"
    plain.write_bytes(compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=activate)).image)
    fused.write_bytes(compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=activate, bias=True)).image)

    left, right = build_operands(_BLOCK_TOKENS, depth, columns, kind="varying", seed=0)
    bias = to_bf16(np.random.default_rng(1).standard_normal(columns).astype(np.float32))
    staged = left.astype(np.float64)
    if activate:
        staged = 0.5 * staged * (1.0 + np.vectorize(math.erf)(staged / math.sqrt(2.0)))
    expected = staged @ right.astype(np.float64)

    plain_error = np.linalg.norm(_run_gemm(plain, left, right, None) - expected) / np.linalg.norm(expected)
    fused_product = _run_gemm(fused, left, right, bias)
    fused_error = np.linalg.norm(fused_product - (expected + bias)) / np.linalg.norm(expected + bias)
    assert fused_error < 0.02
    assert abs(fused_error - plain_error) < 0.001
