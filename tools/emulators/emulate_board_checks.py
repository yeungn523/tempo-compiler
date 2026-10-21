#!/usr/bin/env python3
"""Runs the board's own checks on a host, every launch going through the functional emulator.

board_package/board_checks.py is imported with tools/emulators/emulated_runtime.py installed as its `runtime`, so a
check passing here passes the same assertion the board makes against the same host reference on the same images.
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
import gpt2_inference  # noqa: E402

_CHECKS: dict[str, str] = {
    "copy": "check_copy_argument",
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
    parser.add_argument("checks", nargs="*", metavar="check",
                        help=f"Checks to run, every one when none is named: {', '.join(_CHECKS)}.")
    parser.add_argument("--images", help="Image set to launch, board_package/images when omitted.")
    arguments = parser.parse_args()
    unknown = [name for name in arguments.checks if name not in _CHECKS]
    if unknown:
        parser.error(f"unknown check {', '.join(unknown)}, expected one of {', '.join(_CHECKS)}")
    if arguments.images:
        gpt2_inference._IMAGE_DIRECTORY = pathlib.Path(arguments.images).expanduser().resolve()

    passed = True
    for name in arguments.checks or list(_CHECKS):
        started = time.perf_counter()
        passed &= board_checks.run_check(name=name, check=getattr(board_checks, _CHECKS[name]))
        print(f"    {time.perf_counter() - started:.1f} s on the host, {emulated_runtime.LAUNCHES[0]} launches and "
              f"{emulated_runtime.BUNDLES[0]:,} bundles emulated so far")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
