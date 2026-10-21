#!/usr/bin/env python3
"""Provides the board checks `make kernels` runs, each comparing the device against a host reference."""

from __future__ import annotations

from collections.abc import Callable
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The board package's directory."""

sys.path.insert(0, str(HERE))

from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _CORRUPTION_WORD,
    _LAUNCH_REPEATS,
    _MODEL_WIDTH,
    _on_bf16_grid,
    build_operands,
)
from gpt2_inference import (  # noqa: E402
    UIO_NAME,
    KVCache,
    add_bias_resident,
    add_resident,
    decode_attention,
    decode_step,
    download_activations,
    embed,
    full_gemm,
    halt_image,
    layernorm_resident,
    lm_head,
    load_gpt2_weights,
    next_token_logits,
    npu_seconds,
    prefill,
    transformer_block,
    upload_activations,
    upload_bias,
    upload_layernorm_constants,
)
from references import layernorm_reference, transformer_block_reference  # noqa: E402
from runtime import (  # noqa: E402
    MemcpyKind,
    minitpuFree,
    minitpuGetAddressSpace,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
)


_GEMM_ERROR_LIMIT: float = 0.01


_SOFTMAX_ERROR_LIMIT: float = 0.02


_BLOCK_ERROR_LIMIT: float = 0.05
"""Relative error one transformer block may carry against the host reference."""


_PROMPT: tuple[int, ...] = (464, 3139, 286, 4881, 318)
"""Tokens of "The capital of France is", which the prefill check runs through every layer."""


def check_kernel_launch(image_bytes: bytes) -> None:
    """Verifies the LAUNCH path by staging a resident kernel image and starting it from the host.

    Notes:
        The transfer reaches device memory through the copy engine's alias of the kernel aperture, while the
        loader reads the staged image back through the MINITPU's own device-memory port. Only a launch crosses
        both routes, so an aperture whose two addresses resolve to different DRAM surfaces fails here rather
        than in a workload result. Repeating the launch requires the command processor to return to idle.

    Args:
        image_bytes: The assembled image staged and started.

    Raises:
        RuntimeError: If the loader flags an error and the command processor abandons the launch.
        TimeoutError: If a command fails to retire, which is how a core that never halts presents.
    """
    address_space = minitpuGetAddressSpace()

    minitpuSetDevice(UIO_NAME)
    minitpuHostReset()
    try:
        staging = minitpuHostAlloc(len(image_bytes))
        staging.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
        minitpuMemcpy(
            dst=address_space.kernel_base,
            src=staging,
            nbytes=len(image_bytes),
            kind=MemcpyKind.HostToDevice,
        )

        for _ in range(_LAUNCH_REPEATS):
            minitpuLaunchKernel(kbin_addr=address_space.kernel_base, kbin_size=len(image_bytes))
    finally:
        minitpuHostReset()


def check_copy_roundtrip(words: NDArray[np.uint16], *, uio_name: str = "minitpu") -> None:
    """Verifies that a kernel's argument words cross to device memory and return byte for byte unchanged.

    Notes:
        The words travel from the pinned PS-DDR host buffer to the DMEM device aperture and back, so the check
        covers minitpuMemcpy in both directions over the exact payload a kernel is given. Both transfers translate
        the device pointer the same way, so a returned payload establishes transport fidelity rather than the
        address the words occupy in DMEM.

    Args:
        words: The BF16 container words to send through device memory.
        uio_name: The name of the board's UIO device node.

    Raises:
        AssertionError: If the words the device returns differ from the words sent to it.
    """
    minitpuSetDevice(uio_name)
    minitpuHostReset()
    device_address = None
    try:
        source = minitpuHostAlloc(words.nbytes)
        destination = minitpuHostAlloc(words.nbytes)
        source.view.view("<u2")[: words.size] = words
        device_address = minitpuMalloc(words.nbytes)
        minitpuMemcpy(dst=device_address, src=source, nbytes=words.nbytes, kind=MemcpyKind.HostToDevice)

        # Writes the corruption word so a transfer that moves nothing fails. The udmabuf outlives the process and
        # every run reuses its offsets, so an untouched destination still holds the last run's words. The flush
        # carries the word past the cache lines the read-back invalidates.
        destination.view.view("<u2")[: words.size] = _CORRUPTION_WORD & 0xFFFF
        destination.sync_to_device()

        minitpuMemcpy(dst=destination, src=device_address, nbytes=words.nbytes, kind=MemcpyKind.DeviceToHost)
        np.testing.assert_array_equal(
            destination.view.view("<u2")[: words.size], words)
    finally:
        if device_address is not None:
            minitpuFree(device_address)
        minitpuHostReset()


def run_check(name: str, check: Callable[[], str]) -> bool:
    """Runs one check and reports whether it passed.

    Args:
        name: The label the report line carries.
        check: The callable performing the check, which returns the detail its report line carries.

    Returns:
        Whether the check passed.
    """
    try:
        detail = check()
    except Exception as error:  # noqa: BLE001
        print(f"[FAIL] {name}: {error}")
        return False
    print(f"[PASS] {name}: {detail}")
    return True


def check_halt() -> str:
    """Stages the shortest valid program and starts it repeatedly.

    Returns:
        The detail the report line carries.

    Raises:
        RuntimeError: If the loader flags an error and the command processor abandons the launch.
        TimeoutError: If a command fails to retire, which is how a core that never halts presents.
    """
    check_kernel_launch(image_bytes=halt_image())
    return f"staged and launched {_LAUNCH_REPEATS} times"


def check_copy_argument() -> str:
    """Sends one token block's worth of random words through device memory and reads them back.

    Notes:
        The round trip moves bytes and computes nothing, so random words, which reach every bit pattern, test it
        better than any operand a kernel would take.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the payload read back differs from the payload sent.
    """
    words = np.random.default_rng(0).integers(0, 1 << 16, size=_BLOCK_TOKENS * _MODEL_WIDTH, dtype=np.uint16)
    check_copy_roundtrip(words=words)
    return f"{words.nbytes}-byte random payload"


def check_full_gemm(*, tokens: int = 32, depth: int = 768, columns: int = 3072) -> str:
    """Runs a product at model dimensions through its compiled image and verifies it against a host reference.

    Notes:
        Runs two operand sets, either alone being blind. The structured set is exactly representable and pins
        the kernel to the host reference bit for bit, but its token rows are identical and its weights constant
        along the contraction, so a kernel confusing one token or one contraction block for another still
        passes. The varying set differs along every axis, which makes those confusions observable, and carries
        a relative bound because its BF16 partial sums cost exactness.

    Args:
        tokens: Rows the product covers, which must be a whole number of token blocks.
        depth: The contraction, which a compiled image in the manifest must cover together with `columns`.
        columns: Output features, which must be a whole number of tiles.

    Returns:
        The detail the report line carries, naming the shapes, the launch count and the measured time.

    Raises:
        AssertionError: If the compiled image disagrees with the host reference on either operand set.
    """
    launches = tokens // _BLOCK_TOKENS

    left, right = build_operands(tokens, depth, columns, kind="structured", seed=0)

    started = npu_seconds()
    actual = full_gemm(activations=left, weights=right)
    np.testing.assert_allclose(actual, left @ right, rtol=0, atol=0)

    # build_operands runs assert_operands_discriminating on this pair, so a set that stopped
    # varying along an axis fails here rather than passing the check below for free.
    varying_left, varying_right = build_operands(tokens, depth, columns, kind="varying", seed=0)
    varying_actual = full_gemm(activations=varying_left, weights=varying_right)
    elapsed = npu_seconds() - started
    expected = varying_left.astype(np.float64) @ varying_right.astype(np.float64)
    error = float(np.linalg.norm(varying_actual - expected) / np.linalg.norm(expected))
    if error >= _GEMM_ERROR_LIMIT:
        message = (
            f"Unable to verify the product against the host reference on operands that vary along "
            f"every axis. The relative error must stay below {_GEMM_ERROR_LIMIT:.0%}, which is what "
            f"the BF16 partial sums explain, but got {error:.3%}. The structured operands above "
            f"passed, so the kernel agrees on a product whose token rows are identical and whose "
            f"weights are constant along the contraction, and disagrees once they are not."
        )
        raise AssertionError(message)

    return (
        f"[{tokens},{depth}] @ [{depth},{columns}] compiled, {2 * launches} launches in {elapsed:.3f} s on the NPU "
        f"({1e6 * elapsed / (2 * launches):.0f} us each), varying-operand relative error {error:.2%}"
    )


def check_elementwise_add(*, tokens: int = 32, features: int = 768) -> str:
    """Adds two DMEM tensors, and one tensor and a broadcast bias row, on the device and compares against the host.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If either device sum is not the host's, exactly.
    """
    rng = np.random.default_rng(53)
    h_left = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    h_right = _on_bf16_grid(values=rng.standard_normal((tokens, features)).astype(np.float32))
    h_bias = _on_bf16_grid(values=rng.standard_normal(features).astype(np.float32))
    d_left, d_right, d_bias = upload_activations(h_left), upload_activations(h_right), upload_bias(h_bias)
    try:
        started = npu_seconds()
        d_sum = add_resident(d_left, d_right)
        elapsed = npu_seconds() - started
        d_biased = add_bias_resident(d_left, d_bias)
        try:
            actual, biased = download_activations(d_sum), download_activations(d_biased)
        finally:
            d_sum.free()
            d_biased.free()
    finally:
        d_left.free()
        d_right.free()
        d_bias.free()

    for name, device, expected in (("sum", actual, h_left + h_right), ("bias sum", biased, h_left + h_bias)):
        expected = _on_bf16_grid(values=expected)
        assert np.array_equal(device, expected), (
            f"device {name} differs from the host by up to {np.abs(device - expected).max():.3e}")
    return f"[{tokens},{features}] + [{tokens},{features}] in {elapsed * 1e3:.1f} ms on the NPU, and + bias row, exact"


def check_decode_attention(*, context: int = 48, model: int = _MODEL_WIDTH, heads: int = 12) -> str:
    """Prefills a KV cache, appends a decode step's token and attends it over the cache on the device.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the device disagrees with the host reference.
    """
    rng = np.random.default_rng(19)
    tokens = -(-context // _BLOCK_TOKENS) * _BLOCK_TOKENS
    h_prefill = _on_bf16_grid(values=rng.standard_normal((tokens, 3 * model)).astype(np.float32))
    h_step = _on_bf16_grid(values=rng.standard_normal((_BLOCK_TOKENS, 3 * model)).astype(np.float32))
    d_prefill, d_step = upload_activations(h_prefill), upload_activations(h_step)
    cache = KVCache(capacity=tokens + _BLOCK_TOKENS, model=model)
    try:
        cache.extend(d_prefill, tokens=context)
        cache.append(d_step)
        started = npu_seconds()
        attended = decode_attention(d_step, cache, heads=heads)
        elapsed = npu_seconds() - started
        actual = download_activations(attended)[0]
    finally:
        cache.free()
        d_prefill.free()
        d_step.free()

    head = model // heads
    keys = np.concatenate([h_prefill[:context, model:2 * model], h_step[:1, model:2 * model]], axis=0)
    values = np.concatenate([h_prefill[:context, 2 * model:], h_step[:1, 2 * model:]], axis=0)
    expected = np.empty(model, dtype=np.float64)
    for index in range(heads):
        columns = slice(index * head, (index + 1) * head)
        scores = keys[:, columns].astype(np.float64) @ h_step[0, columns].astype(np.float64)
        weights = np.exp(scores - scores.max())
        expected[columns] = (weights / weights.sum()) @ values[:, columns].astype(np.float64)

    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert error < 0.05, f"decode attention relative error {error:.4%} is too large"
    return (f"{heads} heads x 1 query over {cache.length} cached keys, compiled, 1 launch in {elapsed * 1e3:.1f} ms "
            f"on the NPU, relative error {error:.2%}")


def check_layernorm(*, tokens: int = 32) -> str:
    """Runs the compiled LayerNorm over a resident tensor and verifies it against the host reference."""
    rng = np.random.default_rng(0)
    values = _on_bf16_grid(values=(rng.standard_normal((tokens, _MODEL_WIDTH)) * 2.0).astype(np.float32))
    gamma = _on_bf16_grid(values=(1.0 + 0.1 * rng.standard_normal(_MODEL_WIDTH)).astype(np.float32))
    beta = _on_bf16_grid(values=(0.1 * rng.standard_normal(_MODEL_WIDTH)).astype(np.float32))
    d_rows, d_constants = upload_activations(values), upload_layernorm_constants(gamma, beta)
    try:
        d_normed = layernorm_resident(d_rows, d_constants)
        try:
            actual = download_activations(d_normed)
        finally:
            d_normed.free()
    finally:
        d_rows.free()
        minitpuFree(d_constants)
    expected = layernorm_reference(rows=values, gamma=gamma, beta=beta)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _SOFTMAX_ERROR_LIMIT:
        message = (
            f"Unable to verify LayerNorm against the host reference. The relative error must stay "
            f"below {_SOFTMAX_ERROR_LIMIT:.0%}, but got {error:.3%}."
        )
        raise AssertionError(message)
    return f"[{tokens},{_MODEL_WIDTH}] rows compiled, relative error {error:.2%}"


def check_kv_cache(*, tokens: int = 64, model: int = _MODEL_WIDTH) -> str:
    """Fills a KV cache from a prefill's fused projection and one decode step's, and reads it back.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the cache holds other than each token's keys and values, exactly.
    """
    rng = np.random.default_rng(31)
    h_prefill = _on_bf16_grid(values=rng.standard_normal((tokens, 3 * model)).astype(np.float32))
    h_step = _on_bf16_grid(values=rng.standard_normal((_BLOCK_TOKENS, 3 * model)).astype(np.float32))
    d_prefill, d_step = upload_activations(h_prefill), upload_activations(h_step)
    cache = KVCache(capacity=tokens + _BLOCK_TOKENS, model=model)
    try:
        cache.extend(d_prefill)
        cache.append(d_step)
        held = download_activations(cache.tensor)[:cache.length]
    finally:
        cache.free()
        d_prefill.free()
        d_step.free()
    expected = np.concatenate([h_prefill[:, model:], h_step[:1, model:]], axis=0)
    assert np.array_equal(held, expected), (
        f"the cache differs from the tokens' keys and values by up to {np.abs(held - expected).max():.3e}")
    return f"{tokens} prefill tokens and one decode token, [{cache.length},{2 * model}] keys and values, exact"


def check_gpt2_prefill() -> str:
    """Runs a prompt through GPT-2 small's twelve layers and holds each layer to the host reference.

    Notes:
        Each layer is compared against the reference run on that layer's own device input, so one layer's error is
        measured alone rather than compounded over the layers before it.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If a layer exceeds the error bound, or a cache records other than the prompt's length.
    """
    layers, shared = load_gpt2_weights()
    tokens = len(_PROMPT)
    errors: list[float] = []

    def compare(index: int, layer_input: NDArray[np.float32], layer_output: NDArray[np.float32]) -> None:
        """Measures one layer's real rows against the reference run on the same input."""
        expected = transformer_block_reference(hidden=layer_input, weights=layers[index])[:tokens]
        errors.append(_layer_error(actual=layer_output[:tokens], expected=expected, layer_input=layer_input[:tokens]))

    started = npu_seconds()
    _, caches = prefill(np.asarray(_PROMPT), layers, shared, observe=compare)
    elapsed = npu_seconds() - started
    lengths = {cache.length for cache in caches}
    for cache in caches:
        cache.free()
    worst = int(np.argmax(errors))
    assert lengths == {tokens}, f"the caches recorded lengths {sorted(lengths)} for a {tokens}-token prompt"
    assert errors[worst] < _BLOCK_ERROR_LIMIT, (
        f"layer {worst} differs from the host reference by {errors[worst]:.2%}, past {_BLOCK_ERROR_LIMIT:.0%}")
    return (f"{tokens} prompt tokens, {len(layers)} layers in {elapsed:.2f} s on the NPU, worst layer {worst} at "
            f"{errors[worst]:.2%} relative error, caches holding {tokens} tokens each")


def check_gpt2_decode() -> str:
    """Prefills a prompt, decodes the next token through every layer, and holds both to the host reference.

    Notes:
        Each decode layer is compared against the reference run over the prompt's and the new token's device inputs
        to that layer, so its error is measured alone. The two greedy tokens are held to the reference's own.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If a decode layer exceeds the error bound, a cache records the wrong length, or a greedy
            token differs from the reference's.
    """
    layers, shared = load_gpt2_weights()
    tokens = len(_PROMPT)
    prompt_inputs: dict[int, NDArray[np.float32]] = {}
    errors: list[float] = []

    def remember(index: int, layer_input: NDArray[np.float32], _: NDArray[np.float32]) -> None:
        """Keeps each layer's input for the prompt's real tokens."""
        prompt_inputs[index] = layer_input[:tokens]

    def compare(index: int, layer_input: NDArray[np.float32], layer_output: NDArray[np.float32]) -> None:
        """Measures one decode layer against the reference over the prompt and the new token."""
        rows = np.concatenate([prompt_inputs[index], layer_input[None, :]], axis=0)
        expected = transformer_block_reference(hidden=rows, weights=layers[index])[-1]
        errors.append(_layer_error(actual=layer_output, expected=expected, layer_input=layer_input))

    hidden, caches = prefill(np.asarray(_PROMPT), layers, shared, observe=remember)
    try:
        first = int(np.argmax(next_token_logits(hidden[-1], shared)))
        started = npu_seconds()
        second = int(np.argmax(next_token_logits(decode_step(first, layers, shared, caches, observe=compare),
                                                 shared)))
        elapsed = npu_seconds() - started
        lengths = {cache.length for cache in caches}
    finally:
        for cache in caches:
            cache.free()

    # Runs the same greedy steps through the reference alone, the prompt then the prompt and the first token.
    reference = []
    for sequence in (_PROMPT, (*_PROMPT, first)):
        rows = embed(np.asarray(sequence), shared)
        for weights in layers:
            rows = transformer_block_reference(hidden=rows, weights=weights)
        normalised = layernorm_reference(rows=rows[-1:], gamma=shared["ln_f.g"], beta=shared["ln_f.b"])[0]
        reference.append(int(np.argmax(lm_head(normalised, shared))))

    worst = int(np.argmax(errors))
    assert lengths == {tokens + 1}, f"the caches recorded lengths {sorted(lengths)} after one decode step"
    assert errors[worst] < _BLOCK_ERROR_LIMIT, (
        f"decode layer {worst} differs from the host reference by {errors[worst]:.2%}, past {_BLOCK_ERROR_LIMIT:.0%}")
    assert [first, second] == reference, f"greedy tokens {[first, second]} differ from the reference's {reference}"
    return (f"greedy tokens {[first, second]} match the reference, one decode step through {len(layers)} layers in "
            f"{elapsed:.2f} s on the NPU, worst layer {worst} at {errors[worst]:.2%} relative error")


def check_transformer_block(*, tokens: int = 64) -> str:
    """Runs one GPT-2 transformer block on the device and verifies it against the host reference.

    Notes:
        The whole block runs resident on the device: two LayerNorms, all four projections, every bias and
        residual add, and twelve causal attention heads.

    Args:
        tokens: Sequence length, which the softmax fixes at one VREG's width.

    Returns:
        The detail the report line carries.

    Raises:
        AssertionError: If the relative error exceeds what a block's chained BF16 steps explain.
    """
    rng = np.random.default_rng(1)
    hidden = _on_bf16_grid(values=(rng.standard_normal((tokens, _MODEL_WIDTH)) * 0.5).astype(np.float32))
    weights = _block_weights(model=_MODEL_WIDTH, seed=2)

    started = npu_seconds()
    actual = transformer_block(hidden=hidden, weights=weights)
    elapsed = npu_seconds() - started
    expected = transformer_block_reference(hidden=hidden, weights=weights)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if error >= _BLOCK_ERROR_LIMIT:
        message = (
            f"Unable to verify one transformer block against the host reference. The relative error "
            f"must stay below {_BLOCK_ERROR_LIMIT:.0%}, which is what two LayerNorms, thirteen "
            f"products and twelve softmaxes contribute in BF16, but got {error:.3%}."
        )
        raise AssertionError(message)
    return (f"[{tokens},{_MODEL_WIDTH}] block, 12 heads, in {elapsed * 1e3:.0f} ms on the NPU, "
            f"relative error {error:.2%}")


def _layer_error(*, actual: NDArray[np.float32], expected: NDArray[np.float32],
                 layer_input: NDArray[np.float32]) -> float:
    """Returns one layer's error against the reference, relative to the larger of its input's and output's norms.

    Notes:
        BF16 rounds every value at a step proportional to its own size, so a layer that cancels a large input, as
        GPT-2's last layer does to the first token's massive activations, keeps an error sized by that input.
    """
    scale = max(np.linalg.norm(layer_input), np.linalg.norm(expected))
    return float(np.linalg.norm(actual - expected) / scale)


def _block_weights(model: int, seed: int) -> dict:
    """Builds one block's parameters at GPT-2 initialisation scales, on the BF16 grid."""
    rng = np.random.default_rng(seed)
    hidden = 4 * model

    def normal(shape: tuple, scale: float) -> NDArray[np.float32]:
        return _on_bf16_grid(values=(rng.standard_normal(shape) * scale).astype(np.float32))

    return {
        "ln_1.g": _on_bf16_grid(values=np.ones(model, dtype=np.float32)),
        "ln_1.b": normal((model,), 0.02),
        "attn.qkv.w": normal((model, 3 * model), 0.02),
        "attn.qkv.b": normal((3 * model,), 0.02),
        "attn.proj.w": normal((model, model), 0.02),
        "attn.proj.b": normal((model,), 0.02),
        "ln_2.g": _on_bf16_grid(values=np.ones(model, dtype=np.float32)),
        "ln_2.b": normal((model,), 0.02),
        "mlp.fc.w": normal((model, hidden), 0.02),
        "mlp.fc.b": normal((hidden,), 0.02),
        "mlp.proj.w": normal((hidden, model), 0.02),
        "mlp.proj.b": normal((model,), 0.02),
    }


def main() -> int:
    """Runs the demo programs on the board and reports each result against a host model.

    Returns:
        Zero when every program matched its expected output, and one when any of them did not.
    """
    outcomes = [
        run_check(name="Halt", check=check_halt),
        run_check(name="COPY round trip", check=check_copy_argument),
        run_check(name="Full GEMM", check=check_full_gemm),
        run_check(name="Elementwise add on device", check=check_elementwise_add),
        run_check(name="Decode attention, all heads", check=check_decode_attention),
        run_check(name="LayerNorm", check=check_layernorm),
        run_check(name="KV cache", check=check_kv_cache),
        run_check(name="Transformer block", check=check_transformer_block),
        run_check(name="GPT-2 prefill", check=check_gpt2_prefill),
        run_check(name="GPT-2 decode", check=check_gpt2_decode),
    ]
    print(f"\n{sum(outcomes)} of {len(outcomes)} checks passed")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
