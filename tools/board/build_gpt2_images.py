#!/usr/bin/env python3
"""Compiles every image a GPT-2 small layer launches and writes them, with their manifest, into the board package.

A board carries neither Triton nor the compiler, so the images are compiled here and travel to it as data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from bf16 import _BLOCK_TOKENS, _ROW_BYTES, _TILE, pack_attention_constants  # noqa: E402
from codegen import (ProgramError, compile_copy, compile_launch, compile_pointwise,  # noqa: E402
                      compile_resident_attention, compile_rows, plan, plan_pointwise, plan_rows, read_pointwise,
                      read_program, read_rows, signature_of)
from codegen.asm import AsmBuilder, AsmKernel  # noqa: E402
from codegen.ir import FlashLaunch, Launch  # noqa: E402
from gpt2_triton import (gpt2_add, gpt2_add_bias, gpt2_attention, gpt2_decode_attention,  # noqa: E402
                         gpt2_layernorm, gpt2_matmul, gpt2_matmul_gelu)
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

_ATTENTION: tuple[tuple[int, int], ...] = ((64, 12),)
"""Tokens and heads of each attention image one layer launches, the key loop's length fixed by the tokens."""

_DECODE_KEYS: tuple[int, ...] = (64, 256, 1024)
"""Cache lengths a decode image is built for, each four times the last up to GPT-2's context. A step uses the shortest
that holds its cache and masks the keys past its length, trading attention over masked keys for fewer images."""

_COPIES: tuple[tuple[int, int], ...] = (
    (2 * _MODEL // _TILE * _BLOCK_TOKENS, _ROW_BYTES),
    (2 * _MODEL // _TILE, _BLOCK_TOKENS * _ROW_BYTES),
)
"""Rows and stride of each copy filling the KV cache, being a token block's keys and values after a prefill and
one token's after a decode step."""

_MANIFEST: str = "manifest.json"
"""Name the manifest is written under, beside the images it describes."""


def main() -> int:
    """Compiles every shape the tables above list and writes the images and the manifest.

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
    for tokens, heads in _ATTENTION:
        image, constants, entry = _compile_attention(tokens=tokens, heads=heads)
        name = f"attention_{tokens}.bin"
        (written / name).write_bytes(image)
        (written / entry["constants"]).write_bytes(constants)
        manifest[name] = entry
        print(f"wrote {written / name}  {len(image)} bytes  key_blocks={entry['key_blocks']} heads={heads}")
    for keys in _DECODE_KEYS:
        image, entry = _compile_decode(keys=keys, heads=12)
        name = f"decode_attention_{keys}.bin"
        (written / name).write_bytes(image)
        manifest[name] = entry
    print(f"wrote {len(_DECODE_KEYS)} decode attention images, {_DECODE_KEYS[0]} to {_DECODE_KEYS[-1]} keys")
    for rows, stride in _COPIES:
        name = f"copy_{rows}x{stride}.bin"
        image = compile_copy(rows=rows, stride=stride)
        (written / name).write_bytes(image)
        manifest[name] = {"kind": "copy", "rows": rows, "stride": stride}
        print(f"wrote {written / name}  {len(image)} bytes")
    # Writes the one-bundle program the board's first check launches, which proves a launch retires at all.
    image = AsmKernel(name="halt", bundles=(AsmBuilder().halt(),)).to_bytes(pad_to=4)
    (written / "halt.bin").write_bytes(image)
    manifest["halt.bin"] = {"kind": "halt"}
    print(f"wrote {written / 'halt.bin'}  {len(image)} bytes")
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


def _compile_attention(*, tokens: int, heads: int) -> tuple[bytes, bytes, dict[str, int | str]]:
    """Compiles GPT-2's causal attention over one sequence length and returns its image, constants and entry.

    Args:
        tokens: Tokens the sequence holds, which fixes the key loop's length and the mask tiles.
        heads: Heads the fused QKV tensor carries, which the one launch walks.

    Returns:
        The image bytes, the constants block a host uploads beside it, and the manifest entry naming that block's
        file.

    Raises:
        ProgramError: If the kernel does not plan as one masked attention launch.
    """
    declared = {parameter.name: parameter.default for parameter in gpt2_attention.params if parameter.is_constexpr}
    matmuls = read_program(gpt2_attention, signature=signature_of(gpt2_attention, operands={}),
                           constexprs=declared | {"M": tokens})
    planned = plan(matmuls, strides={})
    launch = planned.launches[0]
    if len(planned.launches) != 1 or not isinstance(launch, FlashLaunch) or launch.mask is None:
        message = (
            f"Unable to compile GPT-2's attention over {tokens} tokens into one image. It must plan as one masked "
            f"attention launch, but got {len(planned.launches)} launches."
        )
        raise ProgramError(message)
    image = compile_resident_attention(launch, heads=heads, fused_tiles=3 * _MODEL // _TILE)
    constants = pack_attention_constants(initial_maximum=launch.entering, constants=launch.constants,
                                         masks=launch.mask_tiles)
    entry = {"kind": "attention", "tokens": tokens, "heads": heads, "key_blocks": launch.key_blocks,
             "constants": f"attention_{tokens}_constants.bin"}
    return image, constants.tobytes(), entry


def _compile_decode(*, keys: int, heads: int) -> tuple[bytes, dict[str, int | float | list | str]]:
    """Compiles GPT-2's decode attention over one cache length and returns its image and manifest entry.

    Args:
        keys: Keys the cache holds, a whole number of token blocks, which fixes the key loop's length.
        heads: Heads the fused QKV tensor and the cache carry.

    Returns:
        The image bytes, and the manifest entry recording what a host packs the constants block from, its mask
        tiles being the host's to write for each step's length.

    Raises:
        ProgramError: If the kernel does not plan as one masked attention launch.
    """
    declared = {parameter.name: parameter.default for parameter in gpt2_decode_attention.params
                if parameter.is_constexpr}
    matmuls = read_program(gpt2_decode_attention, signature=signature_of(gpt2_decode_attention, operands={}),
                           constexprs=declared | {"N": keys, "LENGTH": keys})
    planned = plan(matmuls, strides={})
    launch = planned.launches[0]
    if len(planned.launches) != 1 or not isinstance(launch, FlashLaunch) or launch.mask is None:
        message = (
            f"Unable to compile GPT-2's decode attention over {keys} keys into one image. It must plan as one masked "
            f"attention launch, but got {len(planned.launches)} launches."
        )
        raise ProgramError(message)
    model_tiles = _MODEL // _TILE
    image = compile_resident_attention(launch, heads=heads, fused_tiles=3 * model_tiles, cache_tiles=2 * model_tiles)
    entry = {"kind": "decode_attention", "keys": keys, "heads": heads, "entering": launch.entering,
             "stated": list(launch.constants)}
    return image, entry


if __name__ == "__main__":
    raise SystemExit(main())
