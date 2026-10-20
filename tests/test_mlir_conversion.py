"""Holds the images the MLIR pipeline builds from each Triton kernel to the ones it builds from the tile op stating the
same computation, at GPT-2 small's shapes and at others."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tools" / "board"))

from build_mlir_images import (_TOOLS, add_kernel, attention_kernel, compile_kernel, gemm_kernel,  # noqa: E402
                               layernorm_kernel, triton_kernel)
from gpt2_triton import (gpt2_add, gpt2_add_bias, gpt2_attention, gpt2_decode_attention, gpt2_layernorm,  # noqa: E402
                         gpt2_matmul, gpt2_matmul_bias, gpt2_matmul_gelu)

pytestmark = pytest.mark.skipif(not (_TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")


def _assert_same_image(converted: str, stated: str, directory: pathlib.Path) -> None:
    """Compiles a Triton kernel's module and a tile op kernel and asserts they build the same image."""
    compile_kernel(converted, image=directory / "converted.bin")
    compile_kernel(stated, image=directory / "stated.bin")
    assert (directory / "converted.bin").read_bytes() == (directory / "stated.bin").read_bytes()


@pytest.mark.parametrize(("depth", "columns", "activate", "bias"),
                         [(768, 2304, False, False), (768, 768, False, True), (3072, 768, True, False),
                          (3072, 768, True, True), (512, 1008, False, False), (128, 16, True, True)])
def test_mlir_converts_the_gemm(depth: int, columns: int, activate: bool, bias: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that each GEMM kernel converts into the product its tile op states, with and without the GELU and the
    bias, at GPT-2's shapes and at shapes it never launches."""
    shape = {"N": columns, "K": depth, "stride_am": depth, "stride_bk": columns, "stride_cm": columns,
             "BLOCK_K": min(256, depth)}
    if bias:
        converted = triton_kernel(gpt2_matmul_bias, **shape, ACTIVATE=activate)
    else:
        converted = triton_kernel(gpt2_matmul_gelu if activate else gpt2_matmul, **shape)
    _assert_same_image(converted, gemm_kernel(depth=depth, columns=columns, activate=activate, bias=bias), tmp_path)


@pytest.mark.parametrize(("features", "broadcast"), [(768, False), (768, True), (3072, True), (4096, False)])
def test_mlir_converts_the_add(features: int, broadcast: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that each sum kernel converts into the sum its tile op states, streamed and broadcast."""
    converted = triton_kernel(gpt2_add_bias if broadcast else gpt2_add, N=features)
    _assert_same_image(converted, add_kernel(features=features, broadcast=broadcast), tmp_path)


@pytest.mark.parametrize("features", [768, 1024])
def test_mlir_converts_the_layernorm(features: int, tmp_path: pathlib.Path) -> None:
    """Verifies that the LayerNorm kernel converts into the normalisation its tile op states."""
    _assert_same_image(triton_kernel(gpt2_layernorm, N=features), layernorm_kernel(features=features), tmp_path)


def test_mlir_conversion_follows_the_block_size(tmp_path: pathlib.Path) -> None:
    """Verifies that a GEMM tiled with another BLOCK_N and BLOCK_K converts into the same product, since the tile op
    states the whole block's product whatever the kernel's tiling."""
    shape = {"N": 768, "K": 768, "stride_am": 768, "stride_bk": 768, "stride_cm": 768, "BLOCK_N": 64, "BLOCK_K": 128}
    _assert_same_image(triton_kernel(gpt2_matmul, **shape), gemm_kernel(depth=768, columns=768, activate=False),
                       tmp_path)


@pytest.mark.parametrize("family", ["gemm", "add", "layernorm"])
def test_mlir_conversion_reads_folded_loops(family: str, tmp_path: pathlib.Path) -> None:
    """Verifies that a kernel whose shape gives its loops one iteration each, which the canonicalizer folds away,
    converts into the same tile op as one whose loops survive."""
    if family == "gemm":
        shape = {"M": 32, "N": 16, "K": 256, "stride_am": 256, "stride_bk": 16, "stride_cm": 16}
        converted = triton_kernel(gpt2_matmul, **shape)
        stated = gemm_kernel(depth=256, columns=16, activate=False)
    elif family == "add":
        converted = triton_kernel(gpt2_add_bias, M=32, N=256)
        stated = add_kernel(features=256, broadcast=True)
    else:
        converted = triton_kernel(gpt2_layernorm, M=32, N=256)
        stated = layernorm_kernel(features=256)
    assert "scf.for" not in converted
    _assert_same_image(converted, stated, tmp_path)


def test_mlir_conversion_refuses_a_transposed_operand(tmp_path: pathlib.Path) -> None:
    """Verifies that a GEMM reading its weights transposed is refused, naming the operand, rather than compiled as if
    the host packed them row-major."""
    shape = {"N": 768, "K": 768, "stride_am": 768, "stride_bk": 1, "stride_bn": 768, "stride_cm": 768}
    with pytest.raises(subprocess.CalledProcessError) as refusal:
        _run_opt(triton_kernel(gpt2_matmul, **shape), tmp_path)
    assert "addresses the weights in a layout the host does not pack it in" in refusal.value.stderr


@pytest.mark.parametrize("tokens", [32, 64, 96])
def test_mlir_converts_causal_attention(tokens: int, tmp_path: pathlib.Path) -> None:
    """Verifies that the causal attention kernel, written for one head, converts into the image walking every head,
    at GPT-2's sequence length and at others."""
    _assert_same_image(triton_kernel(gpt2_attention, M=tokens), attention_kernel(keys=tokens, causal=True), tmp_path)


@pytest.mark.parametrize(("keys", "length"), [(64, 64), (256, 256), (1024, 1024), (256, 100)])
def test_mlir_converts_decode_attention(keys: int, length: int, tmp_path: pathlib.Path) -> None:
    """Verifies that the decode attention kernel converts into the image masking its keys at run time, whatever
    length the kernel states, since the host writes the masks for the cache's own length."""
    converted = triton_kernel(gpt2_decode_attention, N=keys, LENGTH=length)
    _assert_same_image(converted, attention_kernel(keys=keys, causal=False), tmp_path)


def test_mlir_conversion_refuses_unnormalised_attention(tmp_path: pathlib.Path) -> None:
    """Verifies that attention whose output is stored without its division by the sum is refused, naming it."""
    module = triton_kernel(gpt2_attention, M=64)
    lines = module.splitlines()
    divide = next(line for line in lines if "arith.divf" in line and "16x64" in line)
    result, accumulated = divide.split("=")[0].strip(), divide.split("arith.divf")[1].split(",")[0].strip()
    module = "\n".join(line for line in lines if line != divide).replace(result, accumulated)
    with pytest.raises(subprocess.CalledProcessError) as refusal:
        _run_opt(module, tmp_path)
    assert "must store the output divided by its sum" in refusal.value.stderr


def _run_opt(module: str, directory: pathlib.Path) -> None:
    """Runs the conversion alone over a module, raising with its diagnostics when it refuses."""
    source = directory / "kernel.ttir"
    source.write_text(module)
    subprocess.run([str(_TOOLS / "minitpu-opt"), str(source), "--convert-tt-to-minitpu=attention-heads=12"], check=True,
                   capture_output=True, text=True)
