#!/usr/bin/env python3
"""Launches the kernel image the minitpu Triton compiler produced and checks the product it drains.
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_TOLERANCE: float = 0.02
"""Relative error an image is allowed past what the BF16 format costs over its own contraction."""

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from bf16 import _ELEMENT_BYTES, from_bf16_words, reduce_drain  # noqa: E402
from runtime import (  # noqa: E402
    MemcpyKind,
    minitpuFree,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
    read_perf,
)


@dataclass(frozen=True, slots=True)
class Measurement:
    """Holds what one run of an image cost, as the device counted it and as the host timed it.

    Attributes:
        launches: Launches the host issued.
        cycles: PL cycles the launches took, summed over them.
        bundles: Bundles the launches issued, summed over them.
        seconds: Wall clock the whole issue sequence took, the host's copies and polling included.
    """

    launches: int
    cycles: int
    bundles: int
    seconds: float

    def report(self) -> str:
        """Returns the cost as one line, being what the device counted and what the host timed around it."""
        return (f"{self.launches} launches, {self.cycles} cycles, {self.bundles} bundles, "
                f"{1e3 * self.seconds:.2f} ms wall")


def _issued(*, order: str, tokens: int, columns: int) -> list[tuple[int, int]]:
    """Returns every launch a host issues, as its token block and column block, in the order the kernel walks them.

    Args:
        order: The output axes the kernel walks, outermost first, as the compiler packed them.
        tokens: Token blocks the product covers.
        columns: Column blocks one token block is issued against.

    Returns:
        The token block and column block of each launch, in issue order.
    """
    if order.startswith("n"):
        return [(token, column) for column in range(columns) for token in range(tokens)]
    return [(token, column) for token in range(tokens) for column in range(columns)]


def _run_on_device(*, image: bytes,
                   operands: np.lib.npyio.NpzFile) -> tuple[NDArray[np.float32], Measurement]:
    """Copies one packed operand pair to the device, launches the image against it, and sums the drain.

    Args:
        image: The kernel image the host compiler produced, already padded to the loader's boundary.
        operands: The packed operands and the sizes the host derived for them.

    Returns:
        The float32 product and what the run cost.
    """
    activations = operands["activations"]
    weights = operands["weights"]
    drain_bytes = int(operands["drain_bytes"])
    zero_tile_words = int(operands["zero_tile_words"])
    shape = tuple(int(extent) for extent in operands["drain_shape"])
    _, _, _, block_tokens, _ = shape
    block_columns = int(operands["block_columns"])
    drain_words = int(operands["drain_words"])
    issued = _issued(order=str(operands["issue_order"]), tokens=len(activations), columns=len(weights))
    product = np.empty((activations.shape[0] * block_tokens, len(weights) * block_columns), dtype=np.float32)

    minitpuSetDevice("minitpu")
    minitpuHostReset()
    owned: list[int] = []
    cycles = bundles = 0
    try:
        host_image = minitpuHostAlloc(len(image))
        host_image.view[: len(image) // 4] = np.frombuffer(image, dtype="<u4")
        device_image = minitpuMalloc(len(image))
        owned.append(device_image)
        minitpuMemcpy(device_image, host_image, len(image), MemcpyKind.HostToDevice)

        host_weights = minitpuHostAlloc(weights[0].nbytes)
        device_weights = minitpuMalloc(int(operands["weight_bytes"]))
        owned.append(device_weights)

        host_drain = minitpuHostAlloc(drain_bytes)
        host_drain.view.view("<u2")[:drain_words] = 0
        device_drain = minitpuMalloc(drain_bytes)
        owned.append(device_drain)
        minitpuMemcpy(device_drain, host_drain, drain_bytes, MemcpyKind.HostToDevice)

        host_activations = minitpuHostAlloc(activations[0].nbytes)
        device_activations = minitpuMalloc(int(operands["activation_bytes"]))
        owned.append(device_activations)

        # Copies both operands every launch, since the issue order decides which of them repeats and the
        # device holds one of each.
        opened = time.perf_counter()
        for token, column in issued:
            block = activations[token]
            host_activations.view.view("<u2")[: block.size] = block
            minitpuMemcpy(device_activations, host_activations, block.nbytes, MemcpyKind.HostToDevice)
            host_weights.view.view("<u2")[: weights[column].size] = weights[column]
            minitpuMemcpy(device_weights, host_weights, weights[column].nbytes, MemcpyKind.HostToDevice)

            minitpuLaunchKernel(device_image, len(image),
                                args=[device_activations, device_weights, device_drain])
            minitpuMemcpy(host_drain, device_drain, drain_bytes, MemcpyKind.DeviceToHost)

            # Reads the counters here rather than once at the end, since the tile reset zeroes the cycle and
            # bundle counts at every launch and a later read would carry the last launch alone.
            sample = read_perf()
            cycles += sample.cycles
            bundles += sample.instrs

            partials = from_bf16_words(words=host_drain.view.view("<u2")[zero_tile_words:drain_words])
            product[token * block_tokens : (token + 1) * block_tokens,
                    column * block_columns : (column + 1) * block_columns] = reduce_drain(partials, shape=shape)
        elapsed = time.perf_counter() - opened
    finally:
        for address in owned:
            minitpuFree(address)
        minitpuHostReset()
    return product, Measurement(launches=len(issued), cycles=cycles, bundles=bundles, seconds=elapsed)


def _run_fused_on_device(*, image: bytes,
                         operands: np.lib.npyio.NpzFile) -> tuple[NDArray[np.float32], Measurement]:
    """Copies a fused layer's three operands to the device, launches the image per token block, and reads back.

    Both operands are copied every launch, as the plain product's runner does, since the device holds one
    buffer of each and a launch walks the whole of both.

    Args:
        image: The kernel image the host compiler produced, already padded to the loader's boundary.
        operands: The packed operands and the sizes the host derived for them.

    Returns:
        The float32 layer output and what the run cost.
    """
    activations = operands["activations"]
    weights = operands["weights"]
    drain_bytes = int(operands["drain_bytes"])
    zero_tile_words = int(operands["zero_tile_words"])
    drain_words = int(operands["drain_words"])
    _, _, output_tiles, block_tokens, tile = (int(extent) for extent in operands["drain_shape"])
    columns = int(operands["output_columns"])
    blocks = activations.shape[0]
    product = np.empty((blocks * block_tokens, columns), dtype=np.float32)

    minitpuSetDevice("minitpu")
    minitpuHostReset()
    owned: list[int] = []
    cycles = bundles = 0
    try:
        host_image = minitpuHostAlloc(len(image))
        host_image.view[: len(image) // 4] = np.frombuffer(image, dtype="<u4")
        device_image = minitpuMalloc(len(image))
        owned.append(device_image)
        minitpuMemcpy(device_image, host_image, len(image), MemcpyKind.HostToDevice)

        host_weights = minitpuHostAlloc(weights.nbytes)
        host_weights.view.view("<u2")[: weights.size] = weights
        device_weights = minitpuMalloc(int(operands["weight_bytes"]))
        owned.append(device_weights)

        host_drain = minitpuHostAlloc(drain_bytes)
        device_drain = minitpuMalloc(drain_bytes)
        owned.append(device_drain)

        host_activations = minitpuHostAlloc(activations[0].nbytes)
        device_activations = minitpuMalloc(int(operands["activation_bytes"]))
        owned.append(device_activations)

        opened = time.perf_counter()
        for token in range(blocks):
            block = activations[token]
            host_activations.view.view("<u2")[: block.size] = block
            minitpuMemcpy(device_activations, host_activations, block.nbytes, MemcpyKind.HostToDevice)
            minitpuMemcpy(device_weights, host_weights, weights.nbytes, MemcpyKind.HostToDevice)
            # The image reads its clearing tile from the drain pointer, so every launch is handed zeros.
            host_drain.view.view("<u2")[:drain_words] = 0
            minitpuMemcpy(device_drain, host_drain, drain_bytes, MemcpyKind.HostToDevice)

            minitpuLaunchKernel(device_image, len(image),
                                args=[device_activations, device_weights, device_drain])
            minitpuMemcpy(host_drain, device_drain, drain_bytes, MemcpyKind.DeviceToHost)

            sample = read_perf()
            cycles += sample.cycles
            bundles += sample.instrs

            # The sweep drains tile by tile, each one a token block, which the layer's rows read back from.
            drained = from_bf16_words(words=host_drain.view.view("<u2")[zero_tile_words:drain_words])
            product[token * block_tokens : (token + 1) * block_tokens] = drained.reshape(
                output_tiles, block_tokens, tile).transpose(1, 0, 2).reshape(block_tokens, columns)
        elapsed = time.perf_counter() - opened
    finally:
        for address in owned:
            minitpuFree(address)
        minitpuHostReset()
    return product, Measurement(launches=blocks, cycles=cycles, bundles=bundles, seconds=elapsed)


def _run_attention_on_device(*, image: bytes,
                             operands: np.lib.npyio.NpzFile) -> tuple[NDArray[np.float32], Measurement]:
    """Copies an attention block's operands to the device, launches the image per query block, and reads back.

    Args:
        image: The kernel image the host compiler produced, already padded to the loader's boundary.
        operands: The packed operands and the sizes the host derived for them.

    Returns:
        The float32 attended output and what the run cost.
    """
    queries = operands["queries"]
    streamed = operands["operands"]
    drain_bytes = int(operands["drain_bytes"])
    drain_words = int(operands["drain_words"])
    shape = tuple(int(extent) for extent in operands["drain_shape"])
    blocks, head = int(operands["row_blocks"]), int(operands["head"])
    rows = shape[3]
    # One launch stages its own block's queries and the two tiles it enters the key loop from.
    span = queries.size // blocks
    attended = np.empty((blocks * rows, head), dtype=np.float32)

    minitpuSetDevice("minitpu")
    minitpuHostReset()
    owned: list[int] = []
    cycles = bundles = 0
    try:
        host_image = minitpuHostAlloc(len(image))
        host_image.view[: len(image) // 4] = np.frombuffer(image, dtype="<u4")
        device_image = minitpuMalloc(len(image))
        owned.append(device_image)
        minitpuMemcpy(device_image, host_image, len(image), MemcpyKind.HostToDevice)

        host_streamed = minitpuHostAlloc(streamed.nbytes)
        host_streamed.view.view("<u2")[: streamed.size] = streamed
        device_streamed = minitpuMalloc(streamed.nbytes)
        owned.append(device_streamed)
        minitpuMemcpy(device_streamed, host_streamed, streamed.nbytes, MemcpyKind.HostToDevice)

        host_queries = minitpuHostAlloc(span * _ELEMENT_BYTES)
        device_queries = minitpuMalloc(span * _ELEMENT_BYTES)
        owned.append(device_queries)

        host_drain = minitpuHostAlloc(drain_bytes)
        device_drain = minitpuMalloc(drain_bytes)
        owned.append(device_drain)

        opened = time.perf_counter()
        for block in range(blocks):
            staged = queries[block * span : (block + 1) * span]
            host_queries.view.view("<u2")[: staged.size] = staged
            minitpuMemcpy(device_queries, host_queries, staged.nbytes, MemcpyKind.HostToDevice)
            # The image reads the keys and values from their own pointer, advancing it a block at a time itself.
            minitpuMemcpy(device_streamed, host_streamed, streamed.nbytes, MemcpyKind.HostToDevice)

            minitpuLaunchKernel(device_image, len(image),
                                args=[device_queries, device_streamed, device_drain])
            minitpuMemcpy(host_drain, device_drain, drain_bytes, MemcpyKind.DeviceToHost)

            sample = read_perf()
            cycles += sample.cycles
            bundles += sample.instrs

            drained = from_bf16_words(words=host_drain.view.view("<u2")[:drain_words])
            attended[block * rows : (block + 1) * rows] = reduce_drain(drained, shape=shape)
        elapsed = time.perf_counter() - opened
    finally:
        for address in owned:
            minitpuFree(address)
        minitpuHostReset()
    return attended, Measurement(launches=blocks, cycles=cycles, bundles=bundles, seconds=elapsed)


def main() -> int:
    """Launches the compiled image and reports its error against the reference the host computed.

    Returns:
        Zero when the product matched within the tolerance the BF16 accumulate chain allows.

    Raises:
        FileNotFoundError: If the image or its operands are not beside this file.
    """
    image_path = _HERE / "triton_kernel.bin"
    operands_path = _HERE / "triton_operands.npz"
    for path in (image_path, operands_path):
        if not path.is_file():
            message = (
                f"Unable to launch without {path.name}. Both it and its companion must sit beside this file, which "
                f"is where the host compiler's copy step puts them."
            )
            raise FileNotFoundError(message)

    image = image_path.read_bytes()
    operands = np.load(operands_path)
    expected: NDArray[np.float32] = operands["expected"]

    print(f"image    {len(image)} bytes, {len(image) // 16} bundles")
    # An attention block states the key blocks it walks, a fused layer the width it drains, and a plain product
    # the tiles and partials it leaves.
    attention = "key_blocks" in operands.files
    fused = "output_columns" in operands.files
    if attention:
        print(f"attend   {expected.shape[0]} by {expected.shape[1]}, "
              f"{int(operands['key_blocks'])} key blocks in one image per query block")
    elif fused:
        print(f"layer    {expected.shape[0]} by {expected.shape[1]}, both projections and the activation "
              f"between them in one image")
    else:
        blocks, groups, output_tiles = (int(extent) for extent in operands["drain_shape"][:3])
        print(f"product  {expected.shape[0]} by {expected.shape[1]}, {blocks * output_tiles} column tiles in "
              f"{blocks} blocks, draining {groups} each")

    run = _run_attention_on_device if attention else (_run_fused_on_device if fused else _run_on_device)
    product, measured = run(image=image, operands=operands)
    error = float(np.abs(product - expected).max() / np.abs(expected).max())
    # A layer contracts twice, so the format costs more over it than the tolerance one product is allowed. The
    # host measures that cost and ships it, leaving the bar what the image adds on top.
    cost = float(operands["format_cost"]) if "format_cost" in operands.files else 0.0
    verdict = "PASS" if error <= cost + _TOLERANCE else "FAIL"
    print(f"device   {verdict}, relative error {error * 100:.4f}%"
          + (f", of which BF16 over this contraction costs {cost * 100:.4f}%" if cost else ""))
    print(f"cost     {measured.report()}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
