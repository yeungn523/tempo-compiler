#!/usr/bin/env python3
"""Compiles every image a GPT-2 small layer launches and writes them, with their manifest, into the board package.

A board carries neither Triton nor the compiler, so the images are compiled here and travel to it as data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from bf16 import _BLOCK_TOKENS  # noqa: E402
from compiler import (ProgramError, compile_launch, compile_pointwise, compile_rows, plan,  # noqa: E402
                      plan_pointwise, plan_rows, read_pointwise, read_program, read_rows, signature_of)
from compiler.ir import Launch  # noqa: E402
from gpt2_triton import gpt2_add, gpt2_add_bias, gpt2_layernorm, gpt2_matmul, gpt2_matmul_gelu  # noqa: E402
from triton.runtime import JITFunction  # noqa: E402

_MODEL: int = 768
"""GPT-2 small's model width."""

_SHAPES: tuple[tuple[int, int, bool], ...] = (
    (_MODEL, 3 * _MODEL, False),
    (_MODEL, _MODEL, False),
    (_MODEL, 4 * _MODEL, False),
    (4 * _MODEL, _MODEL, True),
)
"""Contraction depth, output columns, and whether the left operand is activated, for each GEMM one layer launches."""

_POINTWISE: tuple[tuple[str, int], ...] = (
    ("add", _MODEL),
    ("add_bias", _MODEL),
    ("add_bias", 3 * _MODEL),
    ("add_bias", 4 * _MODEL),
)
"""Kind and feature width of each pointwise image one layer launches, being its residual adds and its biases."""

_POINTWISE_KERNELS: dict[str, JITFunction] = {"add": gpt2_add, "add_bias": gpt2_add_bias}
"""Kernel each pointwise kind is compiled from."""

_ROWS: tuple[tuple[str, int], ...] = (("layernorm", _MODEL),)
"""Kind and feature width of each row image one layer launches, being its LayerNorms."""

_ROW_KERNELS: dict[str, JITFunction] = {"layernorm": gpt2_layernorm}
"""Kernel each row kind is compiled from."""

_MANIFEST: str = "manifest.json"
"""Name the manifest is written under, beside the images it describes."""


def main() -> int:
    """Compiles every shape in `_SHAPES`, `_POINTWISE` and `_ROWS` and writes the images and the manifest.

    Returns:
        Zero once every image is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_HERE / "board_package" / "images"),
                        help="Directory the images and the manifest are written into.")
    arguments = parser.parse_args()

    written = pathlib.Path(arguments.output).expanduser().resolve()
    written.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for depth, columns, activate in _SHAPES:
        image, entry = _compile_gemm(depth=depth, columns=columns, activate=activate)
        name = _image_name(depth=depth, columns=columns, activate=activate)
        (written / name).write_bytes(image)
        manifest[name] = entry
        print(f"wrote {written / name}  {len(image)} bytes  groups={entry['groups']} "
              f"loop_blocks={entry['loop_blocks']} activate={entry['activate']}")
    for kind, features in _POINTWISE:
        image, entry = _compile_pointwise(kind=kind, features=features)
        name = f"{kind}_{features}.bin"
        (written / name).write_bytes(image)
        manifest[name] = entry
        print(f"wrote {written / name}  {len(image)} bytes  chunks={entry['chunks']}")
    for kind, features in _ROWS:
        image, entry = _compile_rows(kind=kind, features=features)
        name = f"{kind}_{features}.bin"
        (written / name).write_bytes(image)
        manifest[name] = entry
        print(f"wrote {written / name}  {len(image)} bytes  rows={entry['rows']} scalars={entry['scalars']}")
    (written / _MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {written / _MANIFEST}")
    return 0


def _image_name(*, depth: int, columns: int, activate: bool) -> str:
    """Returns the file name the image for one GEMM shape is written under."""
    return f"gemm_{depth}x{columns}{'_gelu' if activate else ''}.bin"


def _compile_gemm(*, depth: int, columns: int, activate: bool) -> tuple[bytes, dict[str, int | bool | str]]:
    """Compiles one GEMM shape and returns its image and the launch fields a host issues it with.

    Args:
        depth: Contraction depth the product covers.
        columns: Output columns the product covers.
        activate: Determines whether the left operand passes through the GELU before it enters the array.

    Returns:
        The image bytes, and the manifest entry recording the planned launch and the drain it writes.

    Raises:
        ProgramError: If the plan pads or repacks an operand, if it spans several images, or if its block covers
            other than the token rows one host launch issues.
    """
    kernel = gpt2_matmul_gelu if activate else gpt2_matmul
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    constexprs = declared | {"N": columns, "K": depth, "stride_am": depth, "stride_bk": columns, "stride_cm": columns}
    matmuls = read_program(kernel, signature=signature_of(kernel, operands={}), constexprs=constexprs)
    planned = plan(matmuls, strides={})

    shape = f"[{depth}, {columns}]"
    if len(planned.launches) != 1 or not isinstance(planned.launches[0], Launch):
        message = (
            f"Unable to compile the {shape} GEMM into one image. A layer issues one plain launch per token block, "
            f"but got a plan of {len(planned.launches)} launches."
        )
        raise ProgramError(message)
    if planned.padding or planned.repack:
        message = (
            f"Unable to compile the {shape} GEMM against the resident layout. The host stages operands where "
            f"they lie, so a plan must neither pad nor repack them, but got {planned.padding} {planned.repack}."
        )
        raise ProgramError(message)
    launch = planned.launches[0]
    if launch.block_m != _BLOCK_TOKENS:
        message = (
            f"Unable to compile the {shape} GEMM for the launch a host issues. One launch covers {_BLOCK_TOKENS} "
            f"token rows, but the plan covers {launch.block_m}."
        )
        raise ProgramError(message)

    image, staging = compile_launch(launch)
    entry = {
        "kind": "gemm",
        "depth": depth,
        "columns": columns,
        "activate": launch.activate,
        "column_tiles": launch.column_tiles,
        "groups": launch.groups,
        "loop_blocks": launch.loop_blocks,
        "drain_words": staging.drain_words,
    }
    return image, entry


def _compile_pointwise(*, kind: str, features: int) -> tuple[bytes, dict[str, int | bool | str]]:
    """Compiles one pointwise kind at one feature width and returns its image and manifest entry.

    Args:
        kind: The pointwise kind, naming its kernel in `_POINTWISE_KERNELS`.
        features: Features one token row holds.

    Returns:
        The image bytes, and the manifest entry recording the kind, the width and the chunks one token block spans.

    Raises:
        ProgramError: If the kernel is not one a pointwise image runs at that width.
    """
    kernel = _POINTWISE_KERNELS[kind]
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    pointwise = read_pointwise(kernel, signature=signature_of(kernel, operands={}),
                               constexprs=declared | {"N": features})
    launch = plan_pointwise(pointwise)
    entry = {"kind": kind, "features": features, "chunks": launch.chunks, "broadcast_right": launch.broadcast_right}
    return compile_pointwise(launch), entry


def _compile_rows(*, kind: str, features: int) -> tuple[bytes, dict[str, int | float | list | str]]:
    """Compiles one row kind at one feature width and returns its image and manifest entry.

    Args:
        kind: The row kind, naming its kernel in `_ROW_KERNELS`.
        features: Features one token row holds.

    Returns:
        The image bytes, and the manifest entry recording the kind, the width, the per-feature rows the constants
        block opens with and the number each broadcast constant after them holds.

    Raises:
        ProgramError: If the kernel is not one a row image runs at that width.
    """
    kernel = _ROW_KERNELS[kind]
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    rows = read_rows(kernel, signature=signature_of(kernel, operands={}), constexprs=declared | {"N": features})
    launch = plan_rows(rows)
    entry = {"kind": kind, "features": features, "rows": list(rows.rows), "scalars": list(launch.scalars)}
    return compile_rows(launch), entry


if __name__ == "__main__":
    raise SystemExit(main())
