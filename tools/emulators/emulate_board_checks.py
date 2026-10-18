#!/usr/bin/env python3
"""Runs the board's own checks on a host, every launch going through the functional emulator.

board_package/board_checks.py is imported unchanged with tools/emulators/emulated_runtime.py installed as its `runtime`, so a
check passing here passes the same assertion the board makes, against the same host reference, on the same images.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "tools" / "emulators"))
sys.path.insert(0, str(_HERE / "board_package"))

import emulated_runtime  # noqa: E402

sys.modules["runtime"] = emulated_runtime

import board_checks  # noqa: E402

_CHECKS: dict[str, str] = {
    "gemm": "check_full_gemm",
    "add": "check_elementwise_add",
    "layernorm": "check_layernorm",
    "kv-cache": "check_kv_cache",
    "decode-attention": "check_decode_attention",
    "block": "check_transformer_block",
}
"""Name each check is selected by on the command line, and the board check it runs."""


def main() -> int:
    """Runs the selected board checks through the emulator and reports each.

    Returns:
        Zero when every selected check passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="*", default=list(_CHECKS), choices=list(_CHECKS),
                        help="Checks to run, every one when none is named.")
    arguments = parser.parse_args()

    passed = True
    for name in arguments.checks:
        started = time.perf_counter()
        passed &= board_checks.run_check(name=name, check=getattr(board_checks, _CHECKS[name]))
        print(f"    {time.perf_counter() - started:.1f} s on the host, {emulated_runtime.LAUNCHES[0]} launches and "
              f"{emulated_runtime.BUNDLES[0]:,} bundles emulated so far")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
