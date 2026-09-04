"""Compiles Triton kernels for minitpu."""

import pathlib
import sys

# The board package is not installed, and `backend` reads the array geometry from it, so its directory must reach
# sys.path before the import below resolves.
_BOARD_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "board_package"
sys.path.insert(0, str(_BOARD_PACKAGE))

from .compile import (  # noqa: E402
    MatMul,
    Plan,
    ProgramError,
    plan,
    read_program,
    signature_of,
    stage,
)

__all__ = ["MatMul", "Plan", "ProgramError", "plan", "read_program", "signature_of", "stage"]
