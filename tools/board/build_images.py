"""Builds the board's image set: every image a GPT-2 small layer launches, compiled through the MLIR pipeline from its
Triton source or, for the images with no computation to state in Triton, from its tile op, with the manifest the
runtime launches each with.

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

from driver import Compiled, compile_kernel  # noqa: E402
from gpt2_triton import (gpt2_add, gpt2_add_bias, gpt2_attention, gpt2_decode_attention,  # noqa: E402
                         gpt2_layernorm, gpt2_matmul_bias)

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

_LAYERNORMS: tuple[int, ...] = (768,)
"""Width of each LayerNorm a GPT-2 small layer launches."""

_ATTENTION_TOKENS: tuple[int, ...] = (64,)
"""Sequence length of each causal attention a GPT-2 small prefill launches."""

_DECODE_KEYS: tuple[int, ...] = (64, 256, 1024)
"""Cache length each decode attention image walks, a step using the shortest that holds its cache and masking the
keys past its length, which trades attention over masked keys for fewer images."""

_HEADS: int = 12
"""Attention heads GPT-2 small's layers carry."""

_HEAD: int = 64
"""Width of one of GPT-2 small's attention heads."""

_COPIES: tuple[tuple[int, int], ...] = ((3072, 32), (96, 1024))
"""Rows and stride of each copy filling the KV cache, being a token block's keys and values after a prefill and one
token's after a decode step."""

_MANIFEST: str = "manifest.json"
"""Name the manifest is written under, beside the images it describes."""


def gemm_name(*, depth: int, columns: int, activate: bool, bias: bool = False) -> str:
    """Returns the name of one GEMM image, which its file and its kernel share."""
    return f"gemm_{depth}x{columns}{'_gelu' if activate else ''}{'_bias' if bias else ''}"


def gemm_kernel(*, depth: int, columns: int, activate: bool, bias: bool = False) -> str:
    """Returns the tempo kernel stating one GEMM image's product as a tile op.

    Args:
        depth: The contraction the product covers.
        columns: The output columns the product covers.
        activate: Determines whether each activation register passes through the GELU entering the array.
        bias: Determines whether the product adds the bias row its fourth argument points to.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"tempo.kernel @{gemm_name(depth=depth, columns=columns, activate=activate, bias=bias)} {{",
        "  %activations = tempo.arg 0",
        "  %weights = tempo.arg 1",
        "  %drain = tempo.arg 2",
        *(["  %bias = tempo.arg 3"] if bias else []),
        f"  tempo.matmul %activations, %weights -> %drain{' bias %bias' if bias else ''} tokens 32 depth {depth} "
        f"columns {columns}{' activate' if activate else ''}",
        "}",
        "",
    ])


def add_kernel(*, features: int, broadcast: bool) -> str:
    """Returns the tempo kernel stating one sum image's addition as a tile op.

    Args:
        features: The width the sum covers.
        broadcast: Determines whether the right operand is one bias row added to every token.

    Returns:
        The kernel's MLIR text.
    """
    name = f"add_{'bias_' if broadcast else ''}{features}"
    return "\n".join([
        f"tempo.kernel @{name} {{",
        "  %left = tempo.arg 0",
        "  %right = tempo.arg 1",
        "  %output = tempo.arg 2",
        f"  tempo.add %left, %right -> %output tokens 32 features {features}{' broadcast' if broadcast else ''}",
        "}",
        "",
    ])


def layernorm_kernel(*, features: int) -> str:
    """Returns the tempo kernel stating one LayerNorm image's normalisation as a tile op.

    Args:
        features: The width each token is normalised over.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"tempo.kernel @layernorm_{features} {{",
        "  %rows = tempo.arg 0",
        "  %constants = tempo.arg 1",
        "  %output = tempo.arg 2",
        f"  tempo.layernorm %rows, %constants -> %output tokens 32 features {features} epsilon 1.0e-05",
        "}",
        "",
    ])


def copy_kernel(*, rows: int, stride: int) -> str:
    """Returns the tempo kernel stating one copy image's move as a tile op.

    Notes:
        A copy has no Triton source, the runtime moving the KV cache's rows without computing on them.

    Args:
        rows: The 32-byte rows the copy moves.
        stride: The bytes between consecutive rows at both ends.

    Returns:
        The kernel's MLIR text.
    """
    return "\n".join([
        f"tempo.kernel @copy_{rows}x{stride} {{",
        "  %source = tempo.arg 0",
        "  %destination = tempo.arg 2",
        f"  tempo.copy %source -> %destination rows {rows} stride {stride}",
        "}",
        "",
    ])


def attention_kernel(*, keys: int, causal: bool) -> str:
    """Returns the tempo kernel stating one attention image as a tile op.

    Args:
        keys: The tokens a causal attention attends to each other, or the cached keys a decode attention walks.
        causal: Determines whether the keys are the queries tensor's own rather than a cache's.

    Returns:
        The kernel's MLIR text.
    """
    name = f"attention_{keys}" if causal else f"decode_attention_{keys}"
    return "\n".join([
        f"tempo.kernel @{name} {{",
        "  %constants = tempo.arg 0",
        *([] if causal else ["  %cache = tempo.arg 1"]),
        "  %queries = tempo.arg 2",
        f"  tempo.attention %constants, %queries{'' if causal else ' cache %cache'} heads {_HEADS} head {_HEAD} "
        f"keys {keys}{' causal' if causal else ''}",
        "}",
        "",
    ])


def build_image_set() -> list[Compiled]:
    """Compiles every image the board's image set carries.

    Returns:
        The compiled images, in the order the manifest lists them.

    Raises:
        ProgramError: If the MLIR pipeline refuses a kernel.
    """
    compiled = []
    for depth, columns, activate in _GEMMS:
        shape = {"N": columns, "K": depth, "stride_am": depth, "stride_bk": columns, "stride_cm": columns}
        compiled.append(compile_kernel(gpt2_matmul_bias, **shape, ACTIVATE=activate))
    compiled += [compile_kernel(gpt2_add_bias if broadcast else gpt2_add, N=features) for features, broadcast in _ADDS]
    compiled += [compile_kernel(gpt2_layernorm, N=features) for features in _LAYERNORMS]
    compiled += [compile_kernel(gpt2_attention, heads=_HEADS, M=tokens) for tokens in _ATTENTION_TOKENS]
    compiled += [compile_kernel(gpt2_decode_attention, heads=_HEADS, N=keys, LENGTH=keys) for keys in _DECODE_KEYS]
    compiled += [compile_kernel(copy_kernel(rows=rows, stride=stride)) for rows, stride in _COPIES]
    # Builds the shortest program the loader takes, an empty kernel lowering to its halt alone.
    compiled.append(compile_kernel("tempo.kernel @halt {\n}\n"))
    return compiled


def main() -> int:
    """Writes the image set and its manifest, replacing every image and constants block the directory held.

    Returns:
        Zero once every image is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_HERE / "board_package" / "images"),
                        help="Directory the images and the manifest are written into.")
    arguments = parser.parse_args()

    output = pathlib.Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiled = build_image_set()

    # Removes what a previous build left, so an image no layer launches any longer leaves the set with its entry.
    for stale in output.glob("*.bin"):
        stale.unlink()
    manifest = {}
    for image in compiled:
        path = image.write(output)
        manifest[image.name] = image.entry
        print(f"wrote {path}  {len(image.image) // 16} bundles")
    (output / _MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output / _MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
