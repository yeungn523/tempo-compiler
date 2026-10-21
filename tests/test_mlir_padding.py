"""Holds the GEMMs the MLIR pipeline pads and splits past GPT-2 small's shapes to a float64 reference, through the
emulator: extents a kernel masks, which the host pads with zeros to whole tiles, and contractions deeper than one VMEM
bank holds, which the image runs in segments."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest
import triton
import triton.language as tl

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from bf16 import _BLOCK_TOKENS, _TILE, _ZERO_TILE_WORDS, from_bf16_words, pack_activations, pack_bias, pack_weights  # noqa: E402
from build_images import attention_kernel, gemm_kernel  # noqa: E402
from driver import TOOLS, ProgramError, compile_kernel, triton_kernel  # noqa: E402
from minitpu_emulator import Machine, read_image, to_bf16  # noqa: E402

pytestmark = pytest.mark.skipif(not (TOOLS / "tempo-opt").exists(), reason="mlir/ is not built")

_ROOT_TWO: tl.constexpr = tl.constexpr(1.4142135623730951)
"""Divisor the exact GELU scales its argument by."""


@triton.jit
def indexed_matmul(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    M: tl.constexpr = 64,
    N: tl.constexpr = 40,
    K: tl.constexpr = 100,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Multiplies the activations by the weights, indexing each tile afresh and masking every axis at its extent."""
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                activations = tl.load(activation_pointer + offs_m[:, None] * K + (k + offs_k)[None, :],
                                      mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
                weights = tl.load(weight_pointer + (k + offs_k)[:, None] * N + offs_n[None, :],
                                  mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
                accumulator += tl.dot(activations, weights)
            tl.store(output_pointer + offs_m[:, None] * N + offs_n[None, :], accumulator,
                     mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def advanced_matmul(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    bias_pointer: "*bf16",
    M: tl.constexpr = 64,
    N: tl.constexpr = 40,
    K: tl.constexpr = 100,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
    ACTIVATE: tl.constexpr = False,
    OTHER: tl.constexpr = 0.0,
    WEIGHT_K: tl.constexpr = 0,
):
    """Multiplies the activations, optionally through the exact GELU, by the weights and adds a bias, advancing
    carried pointers and masking the contraction by what remains of it.

    Notes:
        OTHER fills the masked activations and WEIGHT_K, when set, states the weights' contraction apart from the
        activations', the two being what the refusals below vary.
    """
    offs_k = tl.arange(0, BLOCK_K)
    weight_k = K if WEIGHT_K == 0 else WEIGHT_K
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            activation_pointers = activation_pointer + offs_m[:, None] * K + offs_k[None, :]
            weight_pointers = weight_pointer + offs_k[:, None] * N + offs_n[None, :]
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                activations = tl.load(activation_pointers, mask=offs_k[None, :] < K - k, other=OTHER)
                if ACTIVATE:
                    widened = activations.to(tl.float32)
                    activations = (0.5 * widened * (1.0 + tl.math.erf(widened / _ROOT_TWO))).to(tl.bfloat16)
                weights = tl.load(weight_pointers, mask=(offs_k[:, None] < weight_k - k) & (offs_n[None, :] < N),
                                  other=0.0)
                accumulator += tl.dot(activations, weights)
                activation_pointers += BLOCK_K
                weight_pointers += BLOCK_K * N
            bias = tl.load(bias_pointer + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            tl.store(output_pointer + offs_m[:, None] * N + offs_n[None, :], accumulator + bias[None, :],
                     mask=offs_n[None, :] < N)


@triton.jit
def unmasked_matmul(
    activation_pointer: "*bf16",
    weight_pointer: "*bf16",
    output_pointer: "*fp32",
    N: tl.constexpr = 16,
    K: tl.constexpr = 100,
    BLOCK_K: tl.constexpr = 64,
    WINDOW: tl.constexpr = False,
):
    """Multiplies one block of activations by the weights over a contraction its last iteration overruns, unmasked or
    masked by a window that stays where the first iteration reads."""
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((32, N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        pointers = activation_pointer + offs_m[:, None] * K + (k + offs_k)[None, :]
        if WINDOW:
            activations = tl.load(pointers, mask=offs_k[None, :] < K, other=0.0)
        else:
            activations = tl.load(pointers)
        accumulator += tl.dot(activations, tl.load(weight_pointer + (k + offs_k)[:, None] * N + offs_n[None, :]))
    tl.store(output_pointer + offs_m[:, None] * N + offs_n[None, :], accumulator)


def _run_gemm(image: pathlib.Path, left: np.ndarray, right: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Runs one GEMM image over a block of operands at their own extents, which the packers pad to whole tiles, and
    returns the whole padded product it drains."""
    columns = -(-right.shape[1] // _TILE) * _TILE
    activations, weights = pack_activations(activations=left), pack_weights(weights=right)
    bias_words = pack_bias(bias=bias) if bias is not None else np.zeros(0, dtype=np.uint16)
    drain_words = _ZERO_TILE_WORDS + columns // _TILE * _BLOCK_TOKENS * _TILE
    activation_address = 4096
    weight_address = activation_address + activations.nbytes + 4096
    drain_address = weight_address + weights.nbytes + 4096
    bias_address = drain_address + 2 * drain_words + 4096
    arguments = (activation_address, weight_address, drain_address)
    machine = Machine(np.zeros(bias_address + bias_words.nbytes + 4096, dtype=np.uint8),
                      arguments=arguments if bias is None else (*arguments, bias_address))
    machine.write_words(activation_address, activations)
    machine.write_words(weight_address, weights)
    machine.write_words(bias_address, bias_words)
    machine.run(read_image(image))
    tiles = from_bf16_words(words=machine.read_words(drain_address, drain_words)[_ZERO_TILE_WORDS:])
    return tiles.reshape(columns // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)


def _gelu(values: np.ndarray) -> np.ndarray:
    """Returns the exact GELU of BF16 values, rounded back to BF16 as the staging leaves them."""
    widened = values.astype(np.float64)
    return to_bf16((0.5 * widened * (1.0 + np.vectorize(math.erf)(widened / math.sqrt(2.0)))).astype(np.float32))


def _refusal(module: str) -> str:
    """Compiles a module through every stage, expecting a refusal, and returns its diagnostics."""
    with pytest.raises(ProgramError) as refusal:
        compile_kernel(module)
    return str(refusal.value)


@pytest.mark.parametrize(("tokens", "columns", "depth", "block_k"),
                         [(64, 40, 100, 64), (32, 24, 50, 64), (64, 200, 300, 128), (32, 16, 4100, 256)])
def test_mlir_pads_masked_extents(tokens: int, columns: int, depth: int, block_k: int, tmp_path: pathlib.Path) -> None:
    """Verifies that a GEMM masking extents no whole number of tiles covers converts into the product over the padded
    tiles, matching the reference over its own extents and draining zeros past them, a contraction deeper than one
    bank among them."""
    image = tmp_path / "gemm.bin"
    kernel = triton_kernel(indexed_matmul, M=tokens, N=columns, K=depth, BLOCK_K=block_k)
    image.write_bytes(compile_kernel(kernel).image)

    rng = np.random.default_rng(0)
    left = to_bf16(rng.standard_normal((_BLOCK_TOKENS, depth)).astype(np.float32))
    right = to_bf16((rng.standard_normal((depth, columns)) / math.sqrt(depth)).astype(np.float32))
    product = _run_gemm(image, left, right, None)
    expected = left.astype(np.float64) @ right.astype(np.float64)
    assert np.linalg.norm(product[:, :columns] - expected) / np.linalg.norm(expected) < 0.025
    assert not product[:, columns:].any()


@pytest.mark.parametrize("activate", [False, True])
def test_mlir_pads_a_masked_bias(activate: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that a GEMM advancing carried pointers, masking the contraction by what remains of it and adding a
    masked bias, converts into the padded product plus the bias, with and without the GELU."""
    image = tmp_path / "gemm.bin"
    image.write_bytes(compile_kernel(triton_kernel(advanced_matmul, ACTIVATE=activate)).image)

    rng = np.random.default_rng(1)
    left = to_bf16(rng.standard_normal((_BLOCK_TOKENS, 100)).astype(np.float32))
    right = to_bf16((rng.standard_normal((100, 40)) / 10.0).astype(np.float32))
    bias = to_bf16(rng.standard_normal(40).astype(np.float32))
    staged = _gelu(left) if activate else left
    expected = staged.astype(np.float64) @ right.astype(np.float64) + bias
    product = _run_gemm(image, left, right, bias)
    assert np.linalg.norm(product[:, :40] - expected) / np.linalg.norm(expected) < 0.01


@pytest.mark.parametrize(("depth", "columns", "bias"), [(4096, 32, False), (4112, 48, True), (8192, 32, True)])
def test_mlir_splits_a_deep_contraction_exactly(depth: int, columns: int, bias: bool, tmp_path: pathlib.Path) -> None:
    """Verifies that a contraction deeper than one bank holds, split into segments that each resume from the partial
    product the one before drained, sums every k-tile exactly once, on operands whose every partial sum BF16 holds
    exactly."""
    image = tmp_path / "gemm.bin"
    image.write_bytes(compile_kernel(gemm_kernel(depth=depth, columns=columns, activate=False, bias=bias)).image)

    rng = np.random.default_rng(3)
    left = ((rng.random((_BLOCK_TOKENS, depth)) < 16 / depth) * rng.choice([1.0, 2.0], (_BLOCK_TOKENS, depth)))
    right = rng.choice([-1.0, 0.5, 1.0], (depth, columns))
    offsets = rng.integers(-4, 4, columns).astype(np.float32) if bias else None
    product = _run_gemm(image, left.astype(np.float32), right.astype(np.float32), offsets)
    np.testing.assert_array_equal(product, left @ right + (offsets if bias else 0.0))


@pytest.mark.parametrize(("module", "name"), [
    (lambda: gemm_kernel(depth=11536, columns=16, activate=False), "gemm_11536x16"),
    (lambda: attention_kernel(keys=96, causal=True), "attention_96"),
], ids=["deep_gemm", "long_causal_attention"])
def test_mlir_refuses_an_image_past_iram(module, name: str) -> None:
    """Verifies that an image overflowing the instruction memory is refused, naming the bundles it carries, rather than
    written as one the board cannot load: a contraction whose chunks unroll past it, and a causal attention whose query
    blocks do, each block walking its own number of key blocks."""
    assert f"Unable to fit the {name} image in the instruction memory. It carries" in _refusal(module())


@pytest.mark.parametrize(("module", "message"), [
    (lambda: triton_kernel(advanced_matmul, OTHER=1.0), "both operands must be loads, masked ones filling with zero"),
    (lambda: triton_kernel(advanced_matmul, WEIGHT_K=90), "the masks bounding its contraction must agree"),
    (lambda: triton_kernel(unmasked_matmul), "a loop overrunning its bound along the contraction must mask"),
    (lambda: triton_kernel(unmasked_matmul, WINDOW=True), "masks the activations in a way the conversion is unable"),
], ids=["nonzero_fill", "disagreeing_masks", "unmasked_overrun", "fixed_window"])
def test_mlir_refuses_masks_it_cannot_pad(module, message: str) -> None:
    """Verifies that a mask the host's zero padding does not stand in for is refused, naming what differs: a fill other
    than zero, masks stating two extents for one axis, a loop overrunning its bound unmasked, and a mask that stays in
    place while the loop walks the operand."""
    assert message in _refusal(module())
