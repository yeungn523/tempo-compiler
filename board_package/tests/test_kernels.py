"""Guard the one kernel this package deliberately keeps two copies of."""

from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

import kernels  # noqa: E402
import run_minitpu  # noqa: E402


class DuplicateKernelTest(unittest.TestCase):
    """`run_minitpu.py` is meant to be read and edited, so it carries its own copy.

    That is deliberate -- it is the file someone opens to see what a program looks
    like -- but it went wrong once: `kernels.single_mul_add` kept a DMA row pitch of
    64 bytes after BF16-native halved a row to 32, so every descriptor walked DRAM at
    twice the row size and read every other row. `run_minitpu.py`'s copy had been
    fixed and nothing compared the two, so the difference sat there until the whole
    board suite was run end to end for the first time.

    Two copies is fine. Two copies nothing compares is not.
    """

    def test_the_two_copies_assemble_to_the_same_image(self) -> None:
        self.assertEqual(
            bytes(kernels.single_mul_add.image.to_bytes(pad_to=4)),
            bytes(run_minitpu.bf16_mul_add.image.to_bytes(pad_to=4)),
            "kernels.single_mul_add and run_minitpu.bf16_mul_add have diverged; they are "
            "the same program and one of them is now wrong",
        )

    def test_the_halt_examples_agree_too(self) -> None:
        self.assertEqual(
            bytes(kernels.single_halt.image.to_bytes(pad_to=4)),
            bytes(run_minitpu.halt_example.image.to_bytes(pad_to=4)),
        )


if __name__ == "__main__":
    unittest.main()
