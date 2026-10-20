"""Builds the board's image set with every kernel the MLIR pipeline covers compiled by it from its Triton source rather
than by codegen/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from triton.runtime import JITFunction

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from bf16 import _BLOCK_TOKENS, _TILE, as_bf16_words  # noqa: E402
from codegen import signature_of  # noqa: E402
from codegen.frontend import compile_to_ttir  # noqa: E402
from gpt2_triton import (gpt2_add, gpt2_add_bias, gpt2_layernorm, gpt2_matmul, gpt2_matmul_bias,  # noqa: E402
                         gpt2_matmul_gelu)

_TOOLS: pathlib.Path = _HERE / "mlir" / "build" / "bin"
"""Stores the directory minitpu-opt and minitpu-translate are built into."""

_PIPELINE: str = (
    "builtin.module(convert-tt-to-minitpu,minitpu.kernel(minitpu-tile-to-vector,minitpu-allocate-vmem,minitpu-allocate-registers),"
    "convert-minitpu-to-isa,minitpuisa.program(minitpuisa-pack,minitpuisa-schedule))"
)
"""Passes taking a Triton kernel, or a kernel of tile ops, to a scheduled program."""

_GEMMS: tuple[tuple[int, int, bool], ...] = (
    (768, 2304, False),
    (768, 768, False),
    (768, 3072, False),
    (3072, 768, True),
)
"""Depth, columns, and whether the activations pass through the GELU, for each projection a GPT-2 small layer
launches, every one adding its bias in the same image."""

_ADDS: tuple[tuple[int, bool], ...] = (
    (768, False),
    (768, True),
)
"""Width, and whether the right operand is a broadcast bias row, for each sum the image set carries: a layer's
residual add, and the bias sum the board's add check launches."""

_DROPPED_BIAS_SUMS: tuple[int, ...] = (2304, 3072)
"""Width of each bias sum the image set no longer carries, the projections adding their biases themselves."""

_LAYERNORMS: tuple[int, ...] = (768,)
"""Width of each LayerNorm a GPT-2 small layer launches."""

_ATTENTION_TOKENS: tuple[int, ...] = (64,)
"""Sequence length of each causal attention a GPT-2 small prefill launches."""

_DECODE_KEYS: tuple[int, ...] = (64, 256, 1024)
"""Cache length each decode attention image walks, a step using the shortest that holds its cache."""

_HEADS: int = 12
"""Attention heads GPT-2 small's layers carry."""

_HEAD: int = 64
"""Width of one of GPT-2 small's attention heads."""

_MASKED: float = -1.0e30
"""Score a masked key takes, which no real score reaches, and the maximum each query enters its keys below."""

_COPIES: tuple[tuple[int, int], ...] = ((3072, 32), (96, 1024))
"""Rows and stride of each copy filling the KV cache, being a token block's keys and values after a prefill and one
token's after a decode step."""


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
        f"  minitpu.layernorm %rows, %constants -> %output tokens 32 features {features} epsilon 1.0e-05",
        "}",
        "",
    ])


def copy_kernel(*, rows: int, stride: int) -> str:
    """Returns the minitpu kernel stating one copy image's move as a tile op.

    Notes:
        A copy has no Triton source, the runtime moving the KV cache's rows without computing on them.

    Args:
        rows: The 32-byte rows the copy moves.
        stride: The bytes between consecutive rows at both ends.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"minitpu.kernel @copy_{rows}x{stride} {{",
        "  %source = minitpu.arg 0",
        "  %destination = minitpu.arg 2",
        f"  minitpu.copy %source -> %destination rows {rows} stride {stride}",
        "}",
        "",
    ])


def attention_kernel(*, keys: int, causal: bool) -> str:
    """Returns the minitpu kernel stating one attention image as a tile op.

    Args:
        keys: The tokens a causal attention attends to each other, or the cached keys a decode attention walks.
        causal: Determines whether the keys are the queries tensor's own rather than a cache's.

    Returns:
        The kernel's MLIR text.
    """
    name = f"attention_{keys}" if causal else f"decode_attention_{keys}"
    return "\n".join([
        f"minitpu.kernel @{name} {{",
        "  %constants = minitpu.arg 0",
        *([] if causal else ["  %cache = minitpu.arg 1"]),
        "  %queries = minitpu.arg 2",
        f"  minitpu.attention %constants, %queries{'' if causal else ' cache %cache'} heads {_HEADS} head {_HEAD} "
        f"keys {keys}{' causal' if causal else ''}",
        "}",
        "",
    ])


def attention_constants() -> bytes:
    """Returns the constants block a causal attention image reads: tiles of the entering maximum and zeros, and the
    diagonal mask tile laid out keys by queries, masking each key past its query."""
    tile = (_TILE, _TILE)
    keys, queries = np.indices(tile)
    diagonal = np.where(keys > queries, np.float32(_MASKED), np.float32(0.0))
    tiles = [np.full(tile, _MASKED, dtype=np.float32), np.zeros(tile, dtype=np.float32), diagonal]
    return as_bf16_words(values=np.concatenate(tiles, axis=0)).tobytes()


def triton_kernel(kernel: JITFunction, **bound: int | bool) -> str:
    """Returns the tt module of one Triton kernel at the shape its constexprs fix.

    Args:
        kernel: The function the triton.jit decorator returned.
        **bound: The constexprs bound past the kernel's defaults.

    Returns:
        The module's MLIR text.
    """
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    module = compile_to_ttir(kernel, signature=signature_of(kernel, operands={}), constexprs=declared | bound)
    return module.str_nodebug()


def compile_kernel(kernel: str, *, image: pathlib.Path) -> None:
    """Compiles one kernel through the MLIR pipeline and writes its image.

    Args:
        kernel: The kernel's MLIR text, a Triton kernel's tt module or a minitpu kernel of tile ops.
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
    kernels = []
    for depth, columns, activate in _GEMMS:
        shape = {"N": columns, "K": depth, "stride_am": depth, "stride_bk": columns, "stride_cm": columns}
        kernels.append((gemm_name(depth=depth, columns=columns, activate=activate, bias=True),
                        triton_kernel(gpt2_matmul_bias, **shape, ACTIVATE=activate)))
    kernels += [(f"add_{'bias_' if broadcast else ''}{features}",
                 triton_kernel(gpt2_add_bias if broadcast else gpt2_add, N=features))
                for features, broadcast in _ADDS]
    kernels += [(f"layernorm_{features}", triton_kernel(gpt2_layernorm, N=features)) for features in _LAYERNORMS]
    kernels += [(f"copy_{rows}x{stride}", copy_kernel(rows=rows, stride=stride)) for rows, stride in _COPIES]
    kernels += [(f"attention_{tokens}", attention_kernel(keys=tokens, causal=True)) for tokens in _ATTENTION_TOKENS]
    kernels += [(f"decode_attention_{keys}", attention_kernel(keys=keys, causal=False)) for keys in _DECODE_KEYS]
    # Builds the shortest program the loader takes, an empty kernel lowering to its halt alone.
    kernels += [("halt", "minitpu.kernel @halt {\n}\n")]
    for name, kernel in kernels:
        image = output / f"{name}.bin"
        compile_kernel(kernel, image=image)
        print(f"wrote {image}  {image.stat().st_size // 16} bundles, built by mlir/")

    # Removes every image a layer no longer launches, its bias now added inside the product, and records each fused
    # GEMM under its own kind with the drain the runtime allocates for it.
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dropped = [f"{gemm_name(depth=depth, columns=columns, activate=activate)}.bin" for depth, columns, activate in _GEMMS]
    dropped += [f"add_bias_{features}.bin" for features in _DROPPED_BIAS_SUMS]
    for name in dropped:
        manifest.pop(name, None)
        (output / name).unlink(missing_ok=True)
    for depth, columns, activate in _GEMMS:
        manifest[f"{gemm_name(depth=depth, columns=columns, activate=activate, bias=True)}.bin"] = {
            "kind": "gemm_bias", "depth": depth, "columns": columns, "activate": activate,
            "drain_words": _TILE * _TILE + columns * _BLOCK_TOKENS,
        }
    # Writes each causal attention's constants block beside it, and records the masks a decode step writes as laid out
    # keys by queries.
    for tokens in _ATTENTION_TOKENS:
        constants = f"attention_{tokens}_constants.bin"
        (output / constants).write_bytes(attention_constants())
        manifest[f"attention_{tokens}.bin"] = {"kind": "attention", "tokens": tokens, "heads": _HEADS,
                                               "key_blocks": tokens // _TILE, "constants": constants}
    for keys in _DECODE_KEYS:
        manifest[f"decode_attention_{keys}.bin"] = {"kind": "decode_attention", "keys": keys, "heads": _HEADS,
                                                    "entering": _MASKED, "stated": [], "transposed_masks": True}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
