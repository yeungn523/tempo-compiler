"""Compiles Triton kernels for minitpu."""

import pathlib
import sys

# The board package is not installed, and `backend` reads the array geometry from it, so its directory must reach
# sys.path before the imports below resolve.
_BOARD_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "board_package"
sys.path.insert(0, str(_BOARD_PACKAGE))

from .compile import compile_launch, stage  # noqa: E402
from .ir import MatMul, Plan, ProgramError  # noqa: E402
from .planner import plan  # noqa: E402
from .reader import read_program, signature_of  # noqa: E402

__all__ = ["MatMul", "Plan", "ProgramError", "compile_launch", "plan", "read_program", "signature_of",
           "stage"]
