"""Holds the GEMM the MLIR pipeline builds to the board check's own operands and reference, through the emulator."""

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

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _TILE,
    _ZERO_TILE_WORDS,
    build_operands,
    from_bf16_words,
    pack_activations,
    pack_weights,
)
from build_mlir_images import _TOOLS, compile_kernel, gemm_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image  # noqa: E402

pytestmark = pytest.mark.skipif(not (_TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")


@pytest.mark.parametrize("kind", ["structured", "varying"])
def test_mlir_gemm_matches_the_board_reference(kind: str, tmp_path: pathlib.Path) -> None:
    """Verifies that the MLIR-built 768x768 GEMM is exact on structured operands and within the board's bound on varying
    ones."""
    depth = columns = 768
    image = tmp_path / "gemm.bin"
    compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=False), image=image)

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
