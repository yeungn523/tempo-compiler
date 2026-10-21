"""Provides the run of one image against the RTL's kernel image testbench, shared by every RTL check here."""

from __future__ import annotations

from collections.abc import Sequence
import os
import pathlib
import subprocess
import tempfile

_DEFAULT_RTL: pathlib.Path = pathlib.Path(__file__).resolve().parents[3] / "minitpu-tmp"
"""Checkout `minitpu_rtl` falls back to, being a sibling of this repository."""

_PASS_MARKER: str = "=== TB_KERNEL_IMAGE PASS ==="
"""Line the testbench prints once every round has halted."""

_ARGUMENTS: int = 4
"""Argument windows the testbench stages."""


def run_image(image: bytes, *, arguments: Sequence[bytes], dump_bytes: int, what: str, dump: int = 2,
              offsets: Sequence[int] = (), build_dir: str | None = None, max_cycles: int | None = None) -> bytes:
    """Runs one image against the RTL with the argument windows it reads and returns the window it writes.

    Args:
        image: The image.
        arguments: The bytes each argument window holds before the launch, in launch order.
        dump_bytes: Bytes of the dumped window to return.
        what: What the image computes, for the message a failed round carries.
        dump: The argument window the image writes, which is the one returned.
        offsets: The bytes each argument points past the start of its window, for a tensor whose pointer passes a
            zeroed tile opening it. Missing entries are zero.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one. Naming it keeps
            the build across runs, which is what makes a sweep cheap.
        max_cycles: Cycles the round may take before the testbench reports it timed out, or None for the bound the
            testbench carries.

    Returns:
        The first `dump_bytes` bytes of the dumped window after the launch.

    Raises:
        ValueError: If more argument windows are named than the testbench stages.
        RuntimeError: If the testbench does not report the round halted.
    """
    if len(arguments) > _ARGUMENTS:
        message = f"Unable to stage {len(arguments)} arguments. The testbench stages {_ARGUMENTS} windows."
        raise ValueError(message)
    with tempfile.TemporaryDirectory() as scratch:
        stage = pathlib.Path(scratch)
        (stage / "kernel.bin").write_bytes(image)
        plusargs = [f"+KERNEL_BIN={stage / 'kernel.bin'}", f"+DUMP={stage / 'dump.bin'}", f"+DUMP_BYTES={dump_bytes}",
                    f"+DUMP_ARG={dump}"]
        for index, window in enumerate(arguments):
            path = stage / f"arg{index}.bin"
            path.write_bytes(window)
            plusargs.append(f"+ARG{index}={path}")
        plusargs += [f"+ARG_OFFSET{index}={offset}" for index, offset in enumerate(offsets) if offset]
        if max_cycles is not None:
            plusargs.append(f"+MAX_CYCLES={max_cycles}")

        environment = dict(os.environ)
        if build_dir is not None:
            environment["KERNEL_IMAGE_BUILD_DIR"] = str(pathlib.Path(build_dir).resolve())
        root = _rtl_root()
        run = subprocess.run(["bash", str(root / "tb" / "run_kernel_image.sh"), *plusargs], cwd=root,
                             env=environment, capture_output=True, text=True)
        if _PASS_MARKER not in run.stdout:
            rounds = "\n".join(line for line in run.stdout.splitlines() if "KERNEL_IMAGE_ROUND" in line)
            message = (
                f"Unable to complete {what}. The testbench did not report every round halted.\n"
                f"{rounds or run.stdout[-2000:]}\n{run.stderr[-2000:]}"
            )
            raise RuntimeError(message)
        return (stage / "dump.bin").read_bytes()[:dump_bytes]


def _rtl_root() -> pathlib.Path:
    """Returns the checkout carrying the RTL and its kernel image testbench.

    Raises:
        FileNotFoundError: If the checkout does not carry `tb/run_kernel_image.sh`.
    """
    root = pathlib.Path(os.environ.get("minitpu_rtl", _DEFAULT_RTL)).expanduser()
    if not (root / "tb" / "run_kernel_image.sh").is_file():
        message = (
            f"Unable to find the RTL under {root}. This repository does not vendor it, so set minitpu_rtl to a "
            f"checkout carrying tb/run_kernel_image.sh."
        )
        raise FileNotFoundError(message)
    return root
