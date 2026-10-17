#!/usr/bin/env python3
"""Verify the board-time MINITPU artifacts required for deployment.

Presence is not integrity. A `tar | ssh` deploy once reported success while
writing **every text file in the package as zero bytes**; the bitstream survived
because it was the only large file. The three PL scripts then ran and exited 0
having done nothing, and the first symptom was a board that would not program --
several steps removed from the cause. So this checks that the whole package
arrived, not just that the three board-time blobs are present.
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parents[2]
MINITPU_DIR = HERE / "minitpu"
BITSTREAM = MINITPU_DIR / "minitpu_zcu104.bit.bin"
OVERLAY = MINITPU_DIR / "overlay.dtso"
UDMABUF_MODULE = MINITPU_DIR / "u-dma-buf.ko"

# A whole ZCU104 bitstream is about 18 MiB, so this floor clears the one this package
# ships while any materially cut copy of it falls short.
BITSTREAM_MINIMUM_BYTES = 16 * 1024 * 1024

# The one file that is empty on purpose: a namespace marker.
DELIBERATELY_EMPTY = {"minitpu_runtime/minitpu/__init__.py"}


def require(path: pathlib.Path, *, minimum_bytes: int = 1) -> bool:
    """Report whether the file is present and clears the given size floor.

    A blob large enough to survive a truncating deploy can still arrive cut rather than empty, which a
    non-zero size alone reads as success.
    """
    if not path.is_file():
        print(f"  MISSING: {path.relative_to(HERE)}")
        return False
    size = path.stat().st_size
    if size < minimum_bytes:
        print(f"  TRUNCATED: {path.relative_to(HERE)} ({size} bytes, expected {minimum_bytes} or more)")
        return False
    print(f"  {path.relative_to(HERE)} ({size} bytes)")
    return True


def check_sources_are_not_truncated() -> bool:
    """Every Python source and shell script must carry content.

    A deploy that truncates is indistinguishable from a successful one until
    something runs, and `set -euo pipefail` on an empty script exits 0.
    """
    empty: list[str] = []
    counted = 0
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(HERE).as_posix()
        counted += 1
        if path.stat().st_size == 0 and rel not in DELIBERATELY_EMPTY:
            empty.append(rel)
    if empty:
        print(f"  TRUNCATED ({len(empty)} of {counted} source files are empty):")
        for rel in empty[:10]:
            print(f"    {rel}")
        if len(empty) > 10:
            print(f"    ... and {len(empty) - 10} more")
        return False
    print(f"  {counted} source files, none empty")
    return True


def check_kernels_assemble() -> bool:
    """The strongest cheap statement: the images this package exists to run build.

    Catches a partial file that is non-empty but cut mid-statement, which a size
    check cannot see.
    """
    import subprocess
    import sys

    probe = (
        "import sys; sys.path[:0] = [%r, %r]; "
        "import kernels; "
        "n = len(bytes(kernels.single_halt.image.to_bytes(pad_to=4))); "
        "assert n > 0; print('  kernels assemble (%%d bytes for the halt image)' %% n)"
    ) % (str(HERE), str(HERE / "minitpu_runtime"))
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  KERNELS FAILED TO ASSEMBLE:")
        for line in result.stderr.strip().splitlines()[-6:]:
            print(f"    {line}")
        return False
    print(result.stdout.rstrip())
    return True


def main() -> int:
    print("board-time files:")
    blobs = (require(BITSTREAM, minimum_bytes=BITSTREAM_MINIMUM_BYTES)
             and require(OVERLAY) and require(UDMABUF_MODULE))
    print("\npackage integrity:")
    sources = check_sources_are_not_truncated()
    kernels = check_kernels_assemble() if sources else False
    if not (blobs and sources and kernels):
        return 1
    print("\npackage OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
