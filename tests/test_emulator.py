"""Holds the functional emulator to the board's own checks, and shows those checks fail on a miscompiled image.

Each check runs in a subprocess, because the emulated runtime replaces the module named `runtime` for the whole
interpreter it is installed in.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""


def test_board_checks_pass_on_the_emulator() -> None:
    """Verifies that the add, LayerNorm, KV cache and decode attention board checks pass through the emulator."""
    completed = subprocess.run(
        [sys.executable, str(_HERE / "tools" / "emulators" / "emulate_board_checks.py"), "add", "layernorm",
         "kv-cache", "decode-attention"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("[PASS]") == 4, completed.stdout


def test_board_check_fails_on_a_miscompiled_image(tmp_path: pathlib.Path) -> None:
    """Verifies that the add check fails once the add image subtracts, so a passing check is evidence."""
    script = textwrap.dedent(f"""
        import pathlib, shutil, sys
        sys.path.insert(0, {str(_HERE / "tools" / "emulators")!r})
        sys.path.insert(0, {str(_HERE / "board_package")!r})
        sys.path.insert(0, {str(_HERE)!r})
        import emulated_runtime
        sys.modules["runtime"] = emulated_runtime
        import gpt2_inference, board_checks, minitpu_emulator as isa

        images = pathlib.Path({str(tmp_path)!r})
        shutil.copytree(gpt2_inference._IMAGE_DIRECTORY, images, dirs_exist_ok=True)
        path = images / "add_768.bin"
        data = bytearray(path.read_bytes())
        width = (1 << isa._E_SLOT_W) - 1
        vadd, vsub = isa._OPCODES["vadd"], isa._OPCODES["vsub"]
        for offset in range(0, len(data), 16):
            word = int.from_bytes(data[offset:offset + 16], "little")
            operation = (word >> isa._E_SHIFT) & width
            if operation >> 15 == vadd:
                word ^= (vadd ^ vsub) << (isa._E_SHIFT + 15)
                data[offset:offset + 16] = word.to_bytes(16, "little")
        path.write_bytes(bytes(data))
        gpt2_inference._IMAGE_DIRECTORY = images
        try:
            board_checks.check_elementwise_add()
        except AssertionError:
            sys.exit(0)
        sys.exit("the add check passed on an image that subtracts")
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
