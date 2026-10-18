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
FPGA_DIR = HERE / "fpga"
BITSTREAM = FPGA_DIR / "minitpu_zcu104.bit.bin"
OVERLAY = FPGA_DIR / "overlay.dtso"
UDMABUF_MODULE = FPGA_DIR / "u-dma-buf.ko"
MANIFEST = HERE / "images" / "manifest.json"

# A whole ZCU104 bitstream is about 18 MiB, so this floor clears the one this package
# ships while any materially cut copy of it falls short.
BITSTREAM_MINIMUM_BYTES = 16 * 1024 * 1024

# Files that are empty on purpose.
DELIBERATELY_EMPTY: set[str] = set()


def require(path: pathlib.Path, *, minimum_bytes: int = 1) -> bool:
    """Report whether the file is present and clears the given size floor.

    A blob large enough to survive a truncating deploy can still arrive cut rather than empty, which a
    non-zero size alone reads as success.
    """
    if not path.is_file():
        print(f"  missing: {path.relative_to(HERE)}")
        return False
    size = path.stat().st_size
    if size < minimum_bytes:
        print(f"  truncated: {path.relative_to(HERE)} ({size} bytes, expected {minimum_bytes} or more)")
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
        print(f"  truncated ({len(empty)} of {counted} source files are empty):")
        for rel in empty[:10]:
            print(f"    {rel}")
        if len(empty) > 10:
            print(f"    ... and {len(empty) - 10} more")
        return False
    print(f"  {counted} source files, none empty")
    return True


def check_sources_compile() -> bool:
    """Compiles every Python source, which catches a file that is non-empty but cut mid-statement.

    A size check cannot see a partial file, and the first launch would otherwise be what finds it.
    """
    import py_compile

    broken: list[str] = []
    for path in sorted(HERE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            broken.append(f"{path.relative_to(HERE)}: {error.msg.strip().splitlines()[-1]}")
    if broken:
        print(f"  {len(broken)} source files do not compile:")
        for line in broken[:10]:
            print(f"    {line}")
        return False
    print("  every source file compiles")
    return True


def main() -> int:
    print("board-time files:")
    blobs = (require(BITSTREAM, minimum_bytes=BITSTREAM_MINIMUM_BYTES)
             and require(OVERLAY) and require(UDMABUF_MODULE) and require(MANIFEST))
    print("\npackage integrity:")
    sources = check_sources_are_not_truncated()
    compiles = check_sources_compile() if sources else False
    if not (blobs and sources and compiles):
        return 1
    print("\npackage OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
