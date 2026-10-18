"""Compiles Triton kernels for minitpu."""

import pathlib
import sys

# The board package is not installed, and `backend` reads the array geometry from it, so its directory must reach
# sys.path before the imports below resolve.
_BOARD_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "board_package"
sys.path.insert(0, str(_BOARD_PACKAGE))

from .compile import (compile_copy, compile_launch, compile_pointwise, compile_resident_attention,  # noqa: E402
                      compile_rows, stage)
from .ir import FlashLaunch, MatMul, Plan, PointwiseLaunch, ProgramError, RowLaunch  # noqa: E402
from .planner import plan, plan_pointwise, plan_rows  # noqa: E402
from .reader import read_pointwise, read_program, read_rows, signature_of  # noqa: E402

__all__ = ["FlashLaunch", "MatMul", "Plan", "PointwiseLaunch", "ProgramError", "RowLaunch", "compile_copy",
           "compile_launch", "compile_pointwise", "compile_resident_attention", "compile_rows", "plan",
           "plan_pointwise", "plan_rows", "read_pointwise", "read_program", "read_rows", "signature_of", "stage"]
