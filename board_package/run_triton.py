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

from bf16 import from_bf16_words  # noqa: E402
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
    column_tiles, drained, block_tokens, tile = (int(extent) for extent in operands["drain_shape"])
    drain_words = int(operands["drain_words"])
    product = np.empty((activations.shape[0] * block_tokens, column_tiles * tile), dtype=np.float32)

    minitpuSetDevice("minitpu")
    minitpuHostReset()
    owned: list[int] = []
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
        minitpuMemcpy(device_weights, host_weights, weights.nbytes, MemcpyKind.HostToDevice)

        host_drain = minitpuHostAlloc(drain_bytes)
        host_drain.view.view("<u2")[:drain_words] = 0
        device_drain = minitpuMalloc(drain_bytes)
        owned.append(device_drain)
        minitpuMemcpy(device_drain, host_drain, drain_bytes, MemcpyKind.HostToDevice)

        host_activations = minitpuHostAlloc(activations[0].nbytes)
        device_activations = minitpuMalloc(int(operands["activation_bytes"]))
        owned.append(device_activations)

        for index, block in enumerate(activations):
            host_activations.view.view("<u2")[: block.size] = block
            minitpuMemcpy(device_activations, host_activations, block.nbytes, MemcpyKind.HostToDevice)

            minitpuLaunchKernel(device_image, len(image),
                                args=[device_activations, device_weights, device_drain])
            minitpuMemcpy(host_drain, device_drain, drain_bytes, MemcpyKind.DeviceToHost)

            partials = from_bf16_words(words=host_drain.view.view("<u2")[zero_tile_words:drain_words])

            # Adds the reduction groups in float32. The MXU rounds its 24 bit accumulator to BF16 as a result
            # leaves the array, so every partial crossing a block boundary is BF16, and a kernel asking for a
            # float32 accumulator is served here.
            summed = partials.reshape(column_tiles, drained, block_tokens, tile).sum(axis=1, dtype=np.float32)
            product[index * block_tokens : (index + 1) * block_tokens] = (
                summed.transpose(1, 0, 2).reshape(block_tokens, column_tiles * tile))
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
    tiles, groups = (int(extent) for extent in operands["drain_shape"][:2])
    print(f"product  {expected.shape[0]} by {expected.shape[1]}, {tiles} column tiles draining {groups} each")

    product = _run_on_device(image=image, operands=operands)
    error = float(np.abs(product - expected).max() / np.abs(expected).max())
    verdict = "PASS" if error <= 0.02 else "FAIL"
    print(f"device   {verdict}, relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
