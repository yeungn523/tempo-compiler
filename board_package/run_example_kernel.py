#!/usr/bin/env python3
"""Assembles and runs an editable BF16 multiply-add program on the ZCU104.

Assembles the program from the decorated function below on every run, so editing that function is the whole workflow
and the package ships no pre-built kernel images. Expects the programmable logic to already carry the bitstream,
which `sudo make minitpu` loads once after boot. Emits one 128-bit VLIW bundle per `minitpu.bundle` call, and the
final bundle must be `F:halt`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

# Places both package roots on the path ahead of the imports below, which resolve against them and therefore
# carry the E402 suppression.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from runtime import minitpu  # noqa: E402

from bf16 import as_bf16_words, from_bf16_words  # noqa: E402

_MATRIX_SIZE: int = 32
"""The edge length of the square operands the program multiplies."""

_TILES: int = 4
"""The number of tiles the program streams to cover one operand."""

_TILE_ROWS: int = 16
"""The number of VMEM rows one tile occupies."""

_ROW_BYTES: int = 32
"""The byte size of one VMEM row, which holds one BF16 value in each of the sixteen lanes."""

_TILE_BYTES: int = _TILE_ROWS * _ROW_BYTES
"""The byte size of one tile, which is the DRAM stride between the tiles of one operand."""

_SREG_LEFT: int = 0
"""The scalar register holding the DRAM base address of the left operand."""

_SREG_ROW_STRIDE: int = 1
"""The scalar register holding the DMA row pitch."""

_SREG_RIGHT: int = 2
"""The scalar register holding the DRAM base address of the right operand."""

_SREG_OUTPUT: int = 3
"""The scalar register holding the DRAM base address of the output."""

_ROW_LEFT: int = 0
"""The first VMEM row of the left operand's tile."""

_ROW_RIGHT: int = _TILE_ROWS
"""The first VMEM row of the right operand's tile."""

_ROW_OUTPUT: int = 2 * _TILE_ROWS
"""The first VMEM row of the output tile."""

_VREG_LEFT: int = 0
"""The vector register receiving the left operand."""

_VREG_RIGHT: int = 1
"""The vector register receiving the right operand."""

_VREG_PRODUCT: int = 2
"""The vector register receiving the product."""

_VREG_RESULT: int = 3
"""The vector register receiving the sum that is stored back."""

_MULTIPLY_OFFSET: int = 7
"""The bundle distance from the first load to the multiply, which covers the load's result latency."""

_ADD_OFFSET: int = 13
"""The bundle distance from the first load to the add, which covers the multiply's result latency."""

_STORE_OFFSET: int = 19
"""The bundle distance from the first load to the store, which covers the add's result latency."""

_LOOP_BODY_BUNDLES: int = 20
"""The bundle length of the loop body, which the loop buffer replays once per iteration."""

_LOOP_ITERATIONS: int = 4
"""The number of times the loop body replays to walk one tile."""

_LOAD_SLOT_MASK: int = 3
"""The flush mask covering the two DMA slots the tile prologue issues descriptors on."""

_STORE_SLOT_MASK: int = 1
"""The flush mask covering the single DMA slot the tile epilogue issues its descriptor on."""

_AGU_SHIFT: int = 2
"""The address generator's row shift, which advances the walk by one vector register per iteration."""


@minitpu.asm
def bf16_mul_add() -> None:
    """Assembles a 32x32 BF16 C = A * B + A program over four DMA streamed tiles."""

    def pointer(tile: int, register: int, argument: int) -> int:
        """Binds a pointer register to its launch argument on the first tile and steps it one tile on after that.

        Args:
            tile: The index of the tile currently being emitted.
            register: The scalar register that holds the pointer.
            argument: The index of the launch argument the pointer binds to.

        Returns:
            The encoded scalar instruction word.
        """
        if tile == 0:
            return minitpu.smov_arg(rd=register, arg=argument)
        return minitpu.saddi(rd=register, rs=register, value=_TILE_BYTES)

    bundles: dict[int, dict[str, int]] = {}
    program_counter = 0
    for tile in range(_TILES):
        bundles[program_counter] = {"s": pointer(tile=tile, register=_SREG_LEFT, argument=0)}
        program_counter += 1
        bundles[program_counter] = {"s": minitpu.smovi(rd=_SREG_ROW_STRIDE, value=_ROW_BYTES)}
        program_counter += 1
        bundles[program_counter] = {"s": pointer(tile=tile, register=_SREG_RIGHT, argument=1)}
        program_counter += 1
        bundles[program_counter] = {"s": pointer(tile=tile, register=_SREG_OUTPUT, argument=2)}
        program_counter += 1

        # Holds the pointer updates clear of the descriptors that read them, since the scalar writeback is
        # uninterlocked.
        program_counter += 1

        bundles[program_counter] = {
            "d": minitpu.dma_load(
                slot=0, vmem_row=_ROW_LEFT, rows=_TILE_ROWS, base_sreg=_SREG_LEFT, stride_sreg=_SREG_ROW_STRIDE
            )
        }
        program_counter += 1
        bundles[program_counter] = {
            "d": minitpu.dma_load(
                slot=1, vmem_row=_ROW_RIGHT, rows=_TILE_ROWS, base_sreg=_SREG_RIGHT, stride_sreg=_SREG_ROW_STRIDE
            )
        }
        program_counter += 1

        # Covers exactly the two slots the prologue issued descriptors on. A flush blocks until every bit in its
        # mask reports done, and a slot that never carried a descriptor never reports.
        bundles[program_counter] = {"f": minitpu.flush_slot(mask=_LOAD_SLOT_MASK)}
        program_counter += 1
        bundles[program_counter] = {"f": minitpu.lbegin(hi=_LOOP_ITERATIONS)}
        program_counter += 1

        body = program_counter
        bundles[body] = {"x": minitpu.vld(vreg=_VREG_LEFT, row=_ROW_LEFT, agu=True, shift=_AGU_SHIFT)}
        bundles[body + 1] = {"x": minitpu.vld(vreg=_VREG_RIGHT, row=_ROW_RIGHT, agu=True, shift=_AGU_SHIFT)}
        bundles[body + _MULTIPLY_OFFSET] = {"v": minitpu.vmul(vd=_VREG_PRODUCT, va=_VREG_LEFT, vb=_VREG_RIGHT)}
        bundles[body + _ADD_OFFSET] = {"v": minitpu.vadd(vd=_VREG_RESULT, va=_VREG_PRODUCT, vb=_VREG_LEFT)}
        bundles[body + _STORE_OFFSET] = {
            "x": minitpu.vst(vreg=_VREG_RESULT, row=_ROW_OUTPUT, agu=True, shift=_AGU_SHIFT),
            "f": minitpu.loop_end(),
        }
        program_counter = body + _LOOP_BODY_BUNDLES

        bundles[program_counter] = {
            "d": minitpu.dma_store(
                slot=0, vmem_row=_ROW_OUTPUT, rows=_TILE_ROWS, base_sreg=_SREG_OUTPUT, stride_sreg=_SREG_ROW_STRIDE
            )
        }
        program_counter += 1
        bundles[program_counter] = {"f": minitpu.flush_slot(mask=_STORE_SLOT_MASK)}
        program_counter += 1

    bundles[program_counter] = {"f": minitpu.halt()}
    program_counter += 1

    for index in range(program_counter):
        minitpu.bundle(**bundles.get(index, {}))


def main() -> int:
    """Launches the assembled program against a host reference and reports whether the device matched it.

    Returns:
        Zero once the device result equals the reference exactly.
    """
    left: NDArray[np.float32] = np.full((_MATRIX_SIZE, _MATRIX_SIZE), fill_value=2, dtype=np.float32)
    right: NDArray[np.float32] = np.full_like(left, fill_value=2)
    expected: NDArray[np.float32] = left * right + left

    # Hides host and device allocation, every transfer, and the launch behind one call. Takes C-contiguous BF16
    # container words, every argument before the last an input and the last the output.
    host_left = as_bf16_words(values=left).ravel()
    host_right = as_bf16_words(values=right).ravel()
    host_output = np.empty(left.size, dtype="<u2")
    bf16_mul_add(host_left, host_right, host_output)

    actual = from_bf16_words(words=host_output).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    print(f"PASS: {_MATRIX_SIZE}x{_MATRIX_SIZE} BF16 C=A*B+A, {len(bf16_mul_add.image.bundles)} VLIW bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
