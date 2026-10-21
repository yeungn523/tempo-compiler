"""Holds the halt image the MLIR pipeline builds to the shortest program the loader takes."""

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

from driver import TOOLS, compile_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image  # noqa: E402

pytestmark = pytest.mark.skipif(not (TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")


def test_mlir_halt_is_one_bundle(tmp_path: pathlib.Path) -> None:
    """Verifies that an empty kernel builds a single halt bundle, which the machine retires at once."""
    image = tmp_path / "halt.bin"
    image.write_bytes(compile_kernel("minitpu.kernel @halt {\n}\n").image)
    assert image.stat().st_size == 16

    machine = Machine(np.zeros(4096, dtype=np.uint8), arguments=())
    machine.run(read_image(image))
    assert machine.bundles == 1
