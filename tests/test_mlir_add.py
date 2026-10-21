"""Holds the sums the MLIR pipeline builds to an exact BF16 reference, through the emulator."""

from __future__ import annotations

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

from bf16 import _BLOCK_TOKENS, _TILE, from_bf16_words, pack_activations, pack_bias  # noqa: E402
from build_images import add_kernel  # noqa: E402
from driver import TOOLS, compile_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image, to_bf16  # noqa: E402

pytestmark = pytest.mark.skipif(not (TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")


def _run_sum(image: pathlib.Path, left: np.ndarray, right: np.ndarray, *, broadcast: bool) -> tuple[np.ndarray, int]:
    """Runs one sum image over a block and returns the sum it drains and the cycles it took."""
    features = left.shape[1]
    left_words = pack_activations(activations=left)
    right_words = pack_bias(bias=right) if broadcast else pack_activations(activations=right)
    left_address = 4096
    right_address = left_address + left_words.nbytes + 4096
    output_address = right_address + right_words.nbytes + 4096
    machine = Machine(np.zeros(output_address + left_words.nbytes + 4096, dtype=np.uint8),
                      arguments=(left_address, right_address, output_address))
    machine.write_words(left_address, left_words)
    machine.write_words(right_address, right_words)
    machine.run(read_image(image))
    tiles = from_bf16_words(words=machine.read_words(output_address, left_words.size))
    total = tiles.reshape(features // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, features)
    return total, machine.cycles


@pytest.mark.parametrize(("features", "broadcast"),
                         [(128, False), (128, True), (256, False), (256, True), (768, False), (768, True), (2304, True),
                          (3072, True)])
def test_mlir_add_matches_the_reference(features: int, broadcast: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that each MLIR-built sum is exact against a BF16 reference, with no steady loop, one iteration of it, and
    the widths GPT-2 launches."""
    image = tmp_path / "add.bin"
    image.write_bytes(compile_kernel(add_kernel(features=features, broadcast=broadcast)).image)
    rng = np.random.default_rng(features + broadcast)
    left = to_bf16(rng.standard_normal((_BLOCK_TOKENS, features)).astype(np.float32))
    right = to_bf16(rng.standard_normal(features if broadcast else (_BLOCK_TOKENS, features)).astype(np.float32))

    total, _ = _run_sum(image, left, right, broadcast=broadcast)
    np.testing.assert_array_equal(total, to_bf16(left + right))
