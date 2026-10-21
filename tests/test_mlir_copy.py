"""Holds the copies the MLIR pipeline builds to the rows they must move, through the emulator."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from build_images import copy_kernel  # noqa: E402
from driver import TOOLS, compile_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image  # noqa: E402

pytestmark = pytest.mark.skipif(not (TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")

_ROW_WORDS: int = 16
"""BF16 words one 32-byte row holds."""


def _run_copy(image: pathlib.Path, *, rows: int, stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Runs one copy image over random rows and returns the source words, the destination words and the cycles."""
    span = rows * stride
    source_address = 4096
    destination_address = source_address + span + 4096
    machine = Machine(np.zeros(destination_address + span + 4096, dtype=np.uint8),
                      arguments=(source_address, 0, destination_address))
    source = np.random.default_rng(rows + stride).integers(0, 1 << 16, span // 2, dtype=np.uint16)
    machine.write_words(source_address, source)
    machine.run(read_image(image))
    return source, machine.read_words(destination_address, span // 2), machine.cycles


@pytest.mark.parametrize(("rows", "stride"),
                         [(1, 32), (96, 1024), (128, 32), (129, 32), (300, 64), (640, 32), (3072, 32), (384, 1024)])
def test_mlir_copy_moves_the_rows(rows: int, stride: int, tmp_path: pathlib.Path) -> None:
    """Verifies that each copy writes every row it moves and nothing between them, with one chunk, a partial last
    chunk, the loop over chunk pairs, and an advance past saddi's reach."""
    image = tmp_path / "copy.bin"
    image.write_bytes(compile_kernel(copy_kernel(rows=rows, stride=stride)).image)
    source, destination, _ = _run_copy(image, rows=rows, stride=stride)

    spaced = stride // 2
    moved = np.zeros(destination.size, dtype=bool)
    for row in range(rows):
        moved[row * spaced: row * spaced + _ROW_WORDS] = True
    np.testing.assert_array_equal(destination[moved], source[moved])
    assert not destination[~moved].any()
