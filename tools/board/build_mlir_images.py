"""Builds the board's image set with every kernel the MLIR pipeline covers compiled by it rather than by codegen/.
"""

from __future__ import annotations

import argparse
import json
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
    "convert-minitpu-to-isa,minitpuisa.program(minitpuisa-pack,minitpuisa-schedule))"
)
"""Passes taking a kernel of tile ops to a scheduled program."""

_GEMMS: tuple[tuple[int, int, bool], ...] = (
    (768, 2304, False),
    (768, 768, False),
    (768, 3072, False),
    (3072, 768, True),
)
"""Depth, columns, and whether the activations pass through the GELU, for each GEMM a GPT-2 small layer launches."""

_ADDS: tuple[tuple[int, bool], ...] = (
    (768, False),
    (768, True),
    (2304, True),
    (3072, True),
)
"""Width, and whether the right operand is a broadcast bias row, for each sum a GPT-2 small layer launches."""

_LAYERNORMS: tuple[int, ...] = (768,)
"""Width of each LayerNorm a GPT-2 small layer launches."""


def gemm_name(*, depth: int, columns: int, activate: bool, bias: bool = False) -> str:
    """Returns the name of one GEMM image, which its file and its kernel share."""
    return f"gemm_{depth}x{columns}{'_gelu' if activate else ''}{'_bias' if bias else ''}"


def gemm_kernel(*, depth: int, columns: int, activate: bool, bias: bool = False) -> str:
    """Returns the minitpu kernel stating one GEMM image's product as a tile op.

    Args:
        depth: The contraction the product covers.
        columns: The output columns the product covers.
        activate: Determines whether each activation register passes through the GELU entering the array.
        bias: Determines whether the product adds the bias row its fourth argument points to.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"minitpu.kernel @{gemm_name(depth=depth, columns=columns, activate=activate, bias=bias)} {{",
        "  %activations = minitpu.arg 0",
        "  %weights = minitpu.arg 1",
        "  %drain = minitpu.arg 2",
        *(["  %bias = minitpu.arg 3"] if bias else []),
        f"  minitpu.matmul %activations, %weights -> %drain{' bias %bias' if bias else ''} tokens 32 depth {depth} "
        f"columns {columns}{' activate' if activate else ''}",
        "}",
        "",
    ])


def add_kernel(*, features: int, broadcast: bool) -> str:
    """Returns the minitpu kernel stating one sum image's addition as a tile op.

    Args:
        features: The width the sum covers.
        broadcast: Determines whether the right operand is one bias row added to every token.

    Returns:
        The kernel's MLIR text.
    """
    name = f"add_{'bias_' if broadcast else ''}{features}"
    return "\n".join([
        f"minitpu.kernel @{name} {{",
        "  %left = minitpu.arg 0",
        "  %right = minitpu.arg 1",
        "  %output = minitpu.arg 2",
        f"  minitpu.add %left, %right -> %output tokens 32 features {features}{' broadcast' if broadcast else ''}",
        "}",
        "",
    ])


def layernorm_kernel(*, features: int) -> str:
    """Returns the minitpu kernel stating one LayerNorm image's normalisation as a tile op.

    Args:
        features: The width each token is normalised over.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"minitpu.kernel @layernorm_{features} {{",
        "  %rows = minitpu.arg 0",
        "  %constants = minitpu.arg 1",
        "  %output = minitpu.arg 2",
        f"  minitpu.layernorm %rows, %constants -> %output tokens 32 features {features}",
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
    kernels = [(gemm_name(depth=depth, columns=columns, activate=activate, bias=bias),
                gemm_kernel(depth=depth, columns=columns, activate=activate, bias=bias))
               for depth, columns, activate in _GEMMS for bias in (False, True)]
    kernels += [(f"add_{'bias_' if broadcast else ''}{features}", add_kernel(features=features, broadcast=broadcast))
                for features, broadcast in _ADDS]
    kernels += [(f"layernorm_{features}", layernorm_kernel(features=features)) for features in _LAYERNORMS]
    # Builds the shortest program the loader takes, an empty kernel lowering to its halt alone.
    kernels += [("halt", "minitpu.kernel @halt {\n}\n")]
    for name, kernel in kernels:
        image = output / f"{name}.bin"
        compile_kernel(kernel, image=image)
        print(f"wrote {image}  {image.stat().st_size // 16} bundles, built by mlir/")

    # Records each fused GEMM under its own kind, so a runtime asking for a plain GEMM never launches one, and a set
    # without them leaves the runtime adding each bias in a launch of its own.
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for depth, columns, activate in _GEMMS:
        plain = manifest[f"{gemm_name(depth=depth, columns=columns, activate=activate)}.bin"]
        manifest[f"{gemm_name(depth=depth, columns=columns, activate=activate, bias=True)}.bin"] = {
            **plain, "kind": "gemm_bias",
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
