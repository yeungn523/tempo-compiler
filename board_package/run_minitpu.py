#!/usr/bin/env python3
"""Assemble and run an editable BF16 MINITPU program on the ZCU104.

Run ``sudo make minitpu`` once after boot, then simply run this file:

    python3 run_minitpu.py

The program is assembled from the ``@asm.asm`` function below every time this
file runs: there are no pre-built kernel images in this package. Each
``asm.bundle`` call emits one 128-bit VLIW bundle; the final bundle must be
``F:halt``.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from runtime import minitpu  # noqa: E402

@minitpu.asm
def bf16_mul_add():
    """A 32x32 BF16 ``C = A * B + A`` program, assembled at import time."""
    bundles = {}
    pc = 0
    # BF16-native geometry: one VMEM row is one lane stripe, NUM_LANES x 2 B
    # = 32 B. A tile is 16 rows, so 512 B. These were 64 B / 1024 B when a
    # BF16 value still sat in a 32-bit container.
    row_bytes = 16 * 2
    tile_bytes = 16 * row_bytes
    for tile in range(4):
        bundles[pc] = {"s": minitpu.smov_arg(0, 0) if tile == 0 else minitpu.saddi(0, 0, tile_bytes)}; pc += 1
        bundles[pc] = {"s": minitpu.smovi(1, row_bytes)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(2, 1) if tile == 0 else minitpu.saddi(2, 2, tile_bytes)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(3, 2) if tile == 0 else minitpu.saddi(3, 3, tile_bytes)}; pc += 1
        pc += 1  # Explicit NOP: preserve the verified DMA schedule.
        bundles[pc] = {"d": minitpu.dma_load(0, 0, 16, 0, 1)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_load(1, 16, 16, 2, 1)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1
        bundles[pc] = {"f": minitpu.lbegin(4)}; pc += 1
        bundles[pc] = {"x": minitpu.vld(0, 0, agu=True, shift=2)}
        bundles[pc + 1] = {"x": minitpu.vld(1, 16, agu=True, shift=2)}
        bundles[pc + 7] = {"v": minitpu.vmul(2, 0, 1)}
        bundles[pc + 13] = {"v": minitpu.vadd(3, 2, 0)}
        bundles[pc + 19] = {"x": minitpu.vst(3, 32, agu=True, shift=2), "f": minitpu.loop_end()}; pc += 20
        bundles[pc] = {"d": minitpu.dma_store(0, 32, 16, 3, 1)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
    bundles[pc] = {"f": minitpu.halt()}; pc += 1
    for index in range(pc):
        minitpu.bundle(**bundles.get(index, {}))


# Minimal assembly syntax example. This emits a valid program that immediately
# stops; use it as a starting point for an instruction-sequencing smoke test.
# It is intentionally not launched by the matrix demo below.
@minitpu.asm
def halt_example():
    minitpu.bundle(f=minitpu.halt())


def as_bf16_words(values: np.ndarray) -> np.ndarray:
    """Store each float32 in the MINITPU's native 16-bit BF16 container.

    Rounds to nearest even, matching the MXU's pack_bf16, the accumulator's add
    pipe and the SFU's table, so the host uses the same rounding mode as every
    stage it feeds. Truncating here instead biases every operand by up to 1 ULP
    against the device.
    """
    import numpy as np

    bits = values.astype("<f4", copy=False).view("<u4")
    words = bits >> 16
    words &= 1              # bf16 lsb: the tie-break term for round-half-even
    words += 0x7FFF
    words += bits
    words >>= 16
    return words.astype("<u2")


def from_bf16_words(words: np.ndarray) -> np.ndarray:
    import numpy as np

    return (words.astype("<u4") << 16).view("<f4")


def main() -> int:
    import numpy as np

    M = 32
    a = np.full((M, M), 2, dtype=np.float32)
    b = np.full_like(a, 2)
    expected = a * b + a

    # The callable hides host/device allocation, all transfers, and launch.
    h_src1 = as_bf16_words(a).ravel()
    h_src2 = as_bf16_words(b).ravel()
    h_dest = np.empty(a.size, dtype="<u2")
    bf16_mul_add(h_src1, h_src2, h_dest)

    actual = from_bf16_words(h_dest).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    print(f"PASS: {M}x{M} BF16 C=A*B+A, {len(bf16_mul_add.image.bundles)} VLIW bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
