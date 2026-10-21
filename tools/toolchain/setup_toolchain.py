#!/usr/bin/env python3
"""Fetches and verifies the C++ toolchain mlir/ builds against, as pyproject.toml's [tool.tempo.toolchain] pins it.

The toolchain is Triton's source at the pinned tag, for the tt dialect mlir/ compiles, and the prebuilt LLVM that tag
pins, which Triton publishes. Each is fetched once and held to its pin: the LLVM package to its checksum and its
recorded revision, and the Triton source to the LLVM commit its own cmake/llvm-info.json names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

_LLVM_BUILDS: str = "https://oaitriton.blob.core.windows.net/public/llvm-builds"
"""Location Triton publishes its prebuilt LLVM packages under, as python/build_helpers.py names it."""

_TRITON_REPOSITORY: str = "https://github.com/triton-lang/triton.git"
"""Repository the Triton source is cloned from."""

_CHUNK_BYTES: int = 1 << 20
"""Bytes read per step while a download is hashed."""


def main() -> int:
    """Fetches whatever part of the pinned toolchain is missing, verifies all of it, and prints or runs mlir/'s configure
    line.

    Returns:
        Zero once the toolchain matches every pin and, with --configure, once cmake configures the build directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchains", default=str(pathlib.Path.home() / "toolchains"),
                        help="Directory the Triton source is cloned into.")
    parser.add_argument("--llvm-root", default=str(pathlib.Path.home() / ".triton" / "llvm"),
                        help="Directory the LLVM package is unpacked into, Triton's own default.")
    parser.add_argument("--build-dir", default=str(_HERE / "mlir" / "build"),
                        help="Directory mlir/ is configured into.")
    parser.add_argument("--sanitize", action="store_true",
                        help="Instruments the build with AddressSanitizer, which also checks for leaks at exit.")
    parser.add_argument("--configure", action="store_true",
                        help="Runs the configure line instead of printing it.")
    arguments = parser.parse_args()

    pins = tomllib.loads((_HERE / "pyproject.toml").read_text())["tool"]["tempo"]["toolchain"]
    triton = _fetch_triton(pathlib.Path(arguments.toolchains).expanduser(), tag=pins["triton-tag"])
    _check_triton_pins_llvm(triton, commit=pins["llvm-commit"])
    llvm = _fetch_llvm(pathlib.Path(arguments.llvm_root).expanduser(), package=pins["llvm-package"],
                       sha256=pins["llvm-package-sha256"])
    _check_llvm_revision(llvm, commit=pins["llvm-commit"])

    configure = [
        "cmake", "-G", "Ninja", "-S", str(_HERE / "mlir"), "-B", str(pathlib.Path(arguments.build_dir).resolve()),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++",
        "-DLLVM_USE_LINKER=lld",
        f"-DMLIR_DIR={llvm / 'lib' / 'cmake' / 'mlir'}", f"-DLLVM_DIR={llvm / 'lib' / 'cmake' / 'llvm'}",
        f"-DTRITON_SOURCE_DIR={triton}",
    ]
    if arguments.sanitize:
        configure.append("-DLLVM_USE_SANITIZER=Address")
    # The prebuilt LLVM package ships no llvm-lit, so check-tempo runs the lit the mlir dependency group installs.
    if lit := shutil.which("lit"):
        configure.append(f"-DLLVM_EXTERNAL_LIT={lit}")
    print(f"toolchain matches pyproject.toml: Triton {pins['triton-tag']}, LLVM {pins['llvm-commit'][:12]}")
    if arguments.configure:
        subprocess.run(configure, check=True)
    else:
        print("configure mlir/ with:\n  " + shlex.join(configure))
    return 0


def _fetch_triton(toolchains: pathlib.Path, *, tag: str) -> pathlib.Path:
    """Returns the Triton source checkout at `tag`, cloning it shallowly when absent.

    Args:
        toolchains: The directory the checkout lives in.
        tag: The release tag to check out.

    Returns:
        The checkout's root directory.

    Raises:
        SystemExit: If an existing checkout sits at another revision than the tag.
    """
    checkout = toolchains / f"triton-{tag.removeprefix('v')}"
    if not checkout.exists():
        toolchains.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag, _TRITON_REPOSITORY,
                        str(checkout)], check=True)
    described = subprocess.run(["git", "-C", str(checkout), "describe", "--tags", "--exact-match"],
                               capture_output=True, text=True, check=False).stdout.strip()
    if described != tag:
        message = (
            f"Unable to use the Triton source at {checkout}. It must be checked out at {tag}, which pyproject.toml "
            f"pins, but it is at {described or 'an untagged revision'}. Remove it to have it cloned afresh."
        )
        raise SystemExit(message)
    return checkout


def _check_triton_pins_llvm(triton: pathlib.Path, *, commit: str) -> None:
    """Holds the LLVM commit pyproject.toml pins to the one the Triton source itself pins.

    Args:
        triton: The Triton source checkout.
        commit: The LLVM commit pyproject.toml pins.

    Raises:
        SystemExit: If the two commits differ.
    """
    pinned = json.loads((triton / "cmake" / "llvm-info.json").read_text())["llvm_hash"]
    if pinned != commit:
        message = (
            f"Unable to accept pyproject.toml's LLVM pin. It must be the commit Triton's cmake/llvm-info.json names, "
            f"{pinned}, since the tt dialect is compiled against it, but pyproject.toml pins {commit}."
        )
        raise SystemExit(message)


def _fetch_llvm(llvm_root: pathlib.Path, *, package: str, sha256: str) -> pathlib.Path:
    """Returns the unpacked LLVM package, downloading and checksumming it when absent.

    Args:
        llvm_root: The directory LLVM packages are unpacked into.
        package: The package's name, which is also the directory it unpacks to.
        sha256: The checksum the downloaded archive must carry.

    Returns:
        The package's root directory.

    Raises:
        SystemExit: If the downloaded archive's checksum differs from the pin.
    """
    unpacked = llvm_root / package
    if unpacked.exists():
        return unpacked
    llvm_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=llvm_root, suffix=".tar.gz") as archive:
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{_LLVM_BUILDS}/{package}.tar.gz") as response:
            while chunk := response.read(_CHUNK_BYTES):
                digest.update(chunk)
                archive.write(chunk)
        archive.flush()
        if digest.hexdigest() != sha256:
            message = (
                f"Unable to unpack {package}. Its archive must carry the checksum pyproject.toml pins, {sha256}, but "
                f"carries {digest.hexdigest()}."
            )
            raise SystemExit(message)
        with tarfile.open(archive.name) as contents:
            contents.extractall(llvm_root, filter="data")
    return unpacked


def _check_llvm_revision(llvm: pathlib.Path, *, commit: str) -> None:
    """Holds the unpacked LLVM to the commit it was built from, as its own VCSRevision.h records it.

    Args:
        llvm: The LLVM package's root directory.
        commit: The LLVM commit pyproject.toml pins.

    Raises:
        SystemExit: If the package records another revision.
    """
    revision = (llvm / "include" / "llvm" / "Support" / "VCSRevision.h").read_text()
    if commit not in revision:
        message = (
            f"Unable to use the LLVM at {llvm}. It must be built from {commit}, which pyproject.toml pins, but its "
            f"VCSRevision.h records {revision.strip()}."
        )
        raise SystemExit(message)


if __name__ == "__main__":
    sys.exit(main())
