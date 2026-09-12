#!/usr/bin/env python3
"""Launches the kernel image the minitpu Triton compiler produced and checks the product it drains.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from bf16 import from_bf16_words, reduce_drain  # noqa: E402
from runtime import (  # noqa: E402
    MemcpyKind,
    minitpuFree,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuLaunchKernel,
    minitpuMalloc,
    minitpuMemcpy,
    minitpuSetDevice,
)


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


def _run_on_device(*, image: bytes, operands: np.lib.npyio.NpzFile) -> NDArray[np.float32]:
    """Copies one packed operand pair to the device, launches the image against it, and sums the drain.

    Args:
        image: The kernel image the host compiler produced, already padded to the loader's boundary.
        operands: The packed operands and the sizes the host derived for them.

    Returns:
        The float32 product.
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
        for token, column in issued:
            block = activations[token]
            host_activations.view.view("<u2")[: block.size] = block
            minitpuMemcpy(device_activations, host_activations, block.nbytes, MemcpyKind.HostToDevice)
            host_weights.view.view("<u2")[: weights[column].size] = weights[column]
            minitpuMemcpy(device_weights, host_weights, weights[column].nbytes, MemcpyKind.HostToDevice)

            minitpuLaunchKernel(device_image, len(image),
                                args=[device_activations, device_weights, device_drain])
            minitpuMemcpy(host_drain, device_drain, drain_bytes, MemcpyKind.DeviceToHost)

            partials = from_bf16_words(words=host_drain.view.view("<u2")[zero_tile_words:drain_words])
            product[token * block_tokens : (token + 1) * block_tokens,
                    column * block_columns : (column + 1) * block_columns] = reduce_drain(partials, shape=shape)
    finally:
        for address in owned:
            minitpuFree(address)
        minitpuHostReset()
    return product


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
    blocks, groups, output_tiles = (int(extent) for extent in operands["drain_shape"][:3])
    print(f"product  {expected.shape[0]} by {expected.shape[1]}, {blocks * output_tiles} column tiles in "
          f"{blocks} blocks, draining {groups} each")

    product = _run_on_device(image=image, operands=operands)
    error = float(np.abs(product - expected).max() / np.abs(expected).max())
    verdict = "PASS" if error <= 0.02 else "FAIL"
    print(f"device   {verdict}, relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
