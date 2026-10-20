"""Holds the copies the MLIR pipeline builds to the rows they must move and to the images codegen/ compiles, through
the emulator."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from build_mlir_images import _TOOLS, compile_kernel, copy_kernel  # noqa: E402
from codegen import compile_copy  # noqa: E402
from minitpu_emulator import Machine, read_image  # noqa: E402

pytestmark = pytest.mark.skipif(not (_TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")

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
    compile_kernel(copy_kernel(rows=rows, stride=stride), image=image)
    source, destination, _ = _run_copy(image, rows=rows, stride=stride)

    spaced = stride // 2
    moved = np.zeros(destination.size, dtype=bool)
    for row in range(rows):
        moved[row * spaced: row * spaced + _ROW_WORDS] = True
    np.testing.assert_array_equal(destination[moved], source[moved])
    assert not destination[~moved].any()


@pytest.mark.parametrize(("rows", "stride", "faster"), [(3072, 32, True), (96, 1024, False)])
def test_mlir_copy_beats_codegen(rows: int, stride: int, faster: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that each copy GPT-2 launches writes what codegen/'s image writes in fewer bundles, and a copy of
    several chunks in fewer cycles, its drains running beside the next chunk's load.

    Notes:
        A copy of one chunk issues the same descriptors and flushes as codegen/'s, but reads its destination after
        the first flush, where the packer keeps it, which costs two cycles beside the transfer's wait.
    """
    image = tmp_path / "copy.bin"
    compile_kernel(copy_kernel(rows=rows, stride=stride), image=image)
    baseline = tmp_path / "codegen.bin"
    baseline.write_bytes(compile_copy(rows=rows, stride=stride))

    _, destination, cycles = _run_copy(image, rows=rows, stride=stride)
    _, baseline_destination, baseline_cycles = _run_copy(baseline, rows=rows, stride=stride)
    np.testing.assert_array_equal(destination, baseline_destination)
    assert image.stat().st_size < baseline.stat().st_size
    assert cycles < baseline_cycles if faster else cycles <= baseline_cycles + 2
