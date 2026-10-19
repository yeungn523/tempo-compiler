"""Builds the board's image set with every kernel the MLIR pipeline covers compiled by it rather than by codegen/.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

_TOOLS: pathlib.Path = _HERE / "mlir" / "build" / "bin"
"""Stores the directory minitpu-opt and minitpu-translate are built into."""

_PIPELINE: str = (
    "builtin.module(minitpu.kernel(minitpu-tile-to-vector,minitpu-allocate-vmem,minitpu-allocate-registers),"
    "convert-minitpu-to-isa,minitpuisa.program(minitpuisa-schedule))"
)
"""Passes taking a kernel of tile ops to a scheduled program."""

_GEMMS: tuple[tuple[int, int, bool], ...] = (
    (768, 2304, False),
    (768, 768, False),
    (768, 3072, False),
    (3072, 768, True),
)
"""Depth, columns, and whether the activations pass through the GELU, for each GEMM a GPT-2 small layer launches."""


def gemm_kernel(*, depth: int, columns: int, activate: bool) -> str:
    """Returns the minitpu kernel stating one GEMM image's product as a tile op.

    Args:
        depth: The contraction the product covers.
        columns: The output columns the product covers.
        activate: Determines whether each activation register passes through the GELU entering the array.

    Returns:
        The kernel's MLIR text.
    """
    name = f"gemm_{depth}x{columns}{'_gelu' if activate else ''}"
    return "\n".join([
        f"minitpu.kernel @{name} {{",
        "  %activations = minitpu.arg 0",
        "  %weights = minitpu.arg 1",
        "  %drain = minitpu.arg 2",
        f"  minitpu.matmul %activations, %weights -> %drain tokens 32 depth {depth} columns {columns}"
        f"{' activate' if activate else ''}",
        "}",
        "",
    ])


def compile_kernel(kernel: str, *, image: pathlib.Path) -> None:
    """Compiles one kernel through the MLIR pipeline and writes its image.

    Args:
        kernel: The kernel's MLIR text.
        image: The path the image is written to.

    Raises:
        subprocess.CalledProcessError: If either tool refuses the kernel, carrying its diagnostic.
    """
    with tempfile.TemporaryDirectory() as scratch:
        source = pathlib.Path(scratch) / "kernel.mlir"
        program = pathlib.Path(scratch) / "program.mlir"
        source.write_text(kernel)
        subprocess.run([str(_TOOLS / "minitpu-opt"), str(source), f"-pass-pipeline={_PIPELINE}", "-o", str(program)],
                       check=True)
        subprocess.run([str(_TOOLS / "minitpu-translate"), "--export-minitpu-image", str(program), "-o", str(image)],
                       check=True)


def main() -> int:
    """Copies the board's image set and replaces every image the MLIR pipeline covers.

    Returns:
        Zero once every covered image is replaced.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory the image set is written into.")
    arguments = parser.parse_args()

    output = pathlib.Path(arguments.output).expanduser().resolve()
    images = (_HERE / "board_package" / "images").resolve()

    if output != images:
        shutil.copytree(images, output, dirs_exist_ok=True)
    for depth, columns, activate in _GEMMS:
        image = output / f"gemm_{depth}x{columns}{'_gelu' if activate else ''}.bin"
        compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=activate), image=image)
        print(f"wrote {image}  {image.stat().st_size // 16} bundles, built by mlir/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
