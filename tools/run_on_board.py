#!/usr/bin/env python3
"""Compiles one Triton kernel and runs it on a board, over a single SSH session.

A board erases the home directory and revokes the account's runtime devices whenever no session holds it, so a
package copied by one connection is gone before a second one can use it. Everything therefore travels as one tar
stream into one `ssh`, which unpacks it, grants the devices back, and launches the image without the connection
ever dropping.
"""

from __future__ import annotations

import argparse
import getpass
import io
import pathlib
import subprocess
import sys
import tarfile
import tempfile

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

_ADDRESS: str = "132.236.59."
"""Network the course boards sit on, which the last octet selects one of."""

_FORWARDED: tuple[str, ...] = ("operands", "constexprs", "strides", "dims", "activations", "weights",
                               "up-weight", "down-weight")
"""Flags this passes to the compiler entry point untouched, being the ones that describe the launch. A fused
feed-forward layer names its two projections rather than one weight, so both spellings are carried."""


def compile_kernel(*, kernel: str, forwarded: dict[str, str], written: pathlib.Path) -> pathlib.Path:
    """Emits one kernel's image and operands, returning the image.

    Args:
        kernel: The kernel to compile, as the entry point names it.
        forwarded: Value of each compiler flag this call carries through.
        written: Directory the image and operands are written to.

    Returns:
        The path to the emitted image.

    Raises:
        SystemExit: If the compiler refuses the kernel, whose message it has already printed.
    """
    command = [sys.executable, str(_HERE / "kernel_entry_point.py"), "--kernel", kernel,
               "--output", str(written)]
    for name, value in forwarded.items():
        if value:
            command += [f"--{name}", value]
    if subprocess.run(command, check=False).returncode:
        raise SystemExit("compile failed")
    return written / "matmul_0.bin"


def payload(*, images: dict[str, pathlib.Path]) -> bytes:
    """Returns the tar stream one session unpacks, holding the package and every kernel it runs.

    Notes:
        Each kernel travels under its own name in `kernels/`, which the session copies into the two names
        `run_compiled_kernel` reads. One archive therefore carries a whole sweep, and the package it shares is
        sent once however many kernels ride with it.

    Args:
        images: The emitted image of each kernel, keyed by the name it is reported under.

    Returns:
        The gzipped tar bytes.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(_HERE / "board_package", arcname="board_package")
        for name, image in images.items():
            archive.add(image, arcname=f"board_package/kernels/{name}.bin")
            archive.add(image.with_name("matmul_0_operands.npz"), arcname=f"board_package/kernels/{name}.npz")
    return buffer.getvalue()


def _session(*, privileged: str, target: str, command: str = "", names: tuple[str, ...] = ()) -> str:
    """Returns the script one session runs, which reads its payload from the connection's own input.

    Args:
        privileged: A Makefile target to run under sudo before the launch, or the empty string to run none.
        target: A Makefile target to run in place of the compiled kernels, or the empty string to launch them.
        command: A shell command to run in place of both, or the empty string to run neither.
        names: The kernels to launch in turn, each named as its pair sits in `kernels/`.

    Returns:
        The shell script.
    """
    # Reads the sudo password off the head of the stream rather than the command line, where the board's process
    # list would carry it.
    steps = ["set -e", "read -r SUDO_PASSWORD", "rm -rf board_package", "tar xzf -", "cd board_package"]
    if privileged:
        steps.append(f'printf "%s\\n" "$SUDO_PASSWORD" | sudo -S make {privileged}')
    if command:
        steps.append(command)
    elif target:
        steps.append(f"make {target}")
    else:
        # Keeps going after a kernel that fails, so one refusal leaves the rest of a sweep still measured.
        steps.append("status=0")
        for name in names:
            steps.append(f'echo; echo "=== {name} ==="')
            steps.append(f"cp kernels/{name}.bin triton_kernel.bin")
            steps.append(f"cp kernels/{name}.npz triton_operands.npz")
            steps.append("python3 run_compiled_kernel.py || status=$?")
        steps.append("exit $status")
    return "\n".join(steps)


def main() -> int:
    """Compiles the kernel one call names and runs it on the board that call selects.

    Returns:
        What the board's own run returned.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True, action="append", metavar="FILE:NAME",
                        help="the kernel to compile, as a Python file and the name it is declared under, repeated "
                             "to run several in the one session")
    parser.add_argument("--octet", required=True, help=f"last octet of the board's address, under {_ADDRESS}0/24")
    parser.add_argument("--user", default=getpass.getuser(), help="the account the board authorises, being a netid")
    parser.add_argument("--run", default="", metavar="COMMAND",
                        help="run this shell command on the board instead of the compiled kernel, from inside the "
                             "unpacked package")
    parser.add_argument("--make", default="", metavar="TARGET",
                        help="run this Makefile target on the board instead of the compiled kernel, such as "
                             "kernels for the hand-written board checks")
    parser.add_argument("--program", action="store_true",
                        help="set the PL clock, program the PL and provision the runtime devices before launching")
    parser.add_argument("--no-provision", action="store_true",
                        help="launch without granting the runtime devices first, for a board already holding this "
                             "account's session")
    for name in _FORWARDED:
        parser.add_argument(f"--{name}", default="", help=f"passed to kernel_entry_point.py as --{name}")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        forwarded = {name: getattr(arguments, name.replace("-", "_")) for name in _FORWARDED}
        images = {}
        for kernel in arguments.kernel:
            # Names each pair by the kernel it came from, which one file declaring several leaves after the colon.
            name = kernel.rpartition(":")[2] or pathlib.Path(kernel).stem
            images[name] = compile_kernel(kernel=kernel, forwarded=forwarded,
                                          written=pathlib.Path(scratch) / name)
        stream = payload(images=images)

    # Provisions every run, since a board revokes this account's devices once its session releases, exactly as it
    # erases the home directory. Programming provisions as its own last step.
    privileged = "minitpu" if arguments.program else ("" if arguments.no_provision else "provision")
    password = getpass.getpass("sudo password for the board: ") if privileged else ""
    target = f"{arguments.user}@{_ADDRESS}{arguments.octet}"
    
    print(f"running {len(images)} on {target}, {len(stream) / 1e6:.1f} MB to send")
    return subprocess.run(["ssh", target,
                           _session(privileged=privileged, target=arguments.make, command=arguments.run,
                                    names=tuple(images))],
                          input=password.encode() + b"\n" + stream, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
