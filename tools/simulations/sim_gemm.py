#!/usr/bin/env python3
"""Provides the RTL check that a GEMM image computes the product it claims.

The image is compiled through the MLIR pipeline, either from a Triton kernel named on the command line or from the
tempo.matmul its dimensions state, and runs once per token block in the RTL's kernel image testbench, so a kernel is
checked here before it reaches hardware.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The directory holding the RTL checks."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "board_package"))
sys.path.insert(0, str(_HERE.parent / "board"))

from bf16 import (_BLOCK_TOKENS, _TILE, _ZERO_TILE_WORDS, _on_bf16_grid, from_bf16_words,  # noqa: E402
                  pack_activations, pack_bias, pack_weights)
from build_images import gemm_kernel  # noqa: E402
from driver import Compiled, compile_kernel  # noqa: E402
from rtl import run_image  # noqa: E402
from triton.runtime import JITFunction  # noqa: E402

_TOLERANCE: float = 0.02
"""Relative error the BF16 accumulate chain is allowed against the reference's norm, which the board check also holds
to."""

_DEEP_TOLERANCE: float = 0.03
"""Relative error a contraction deeper than one bank holds is allowed, its BF16 accumulators summing enough terms to
drift past the shallower bound."""

_DEEP: int = 3840
"""Deepest contraction one segment stages, past which the image splits it and the deeper bound applies."""


def load_kernel(specification: str) -> JITFunction:
    """Returns the Triton kernel a specification names.

    Args:
        specification: The kernel, as the path to a Python file and, after a colon, the name it is declared under.

    Raises:
        ValueError: If the specification names no triton.jit function.
    """
    path, _, name = specification.rpartition(":")
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, name, None)
    if not isinstance(kernel, JITFunction):
        message = f"Unable to read {specification} as a Triton kernel. It must name a triton.jit function in {path}."
        raise ValueError(message)
    return kernel


def simulate(compiled: Compiled, *, left: NDArray[np.float32], right: NDArray[np.float32],
             bias: NDArray[np.float32] | None, build_dir: str | None = None,
             max_cycles: int | None = None) -> float:
    """Runs a GEMM image against the RTL once per token block and returns its error against a float64 reference.

    Args:
        compiled: The GEMM image and its manifest entry.
        left: The activations, a whole number of token blocks, at the extent the kernel states.
        right: The weights, at the extent the kernel states.
        bias: The bias row an image adding one reads, or None.
        build_dir: Directory the Verilator build is kept in, or None to build in a temporary one.
        max_cycles: Cycles one round may take, or None for the bound the testbench carries.

    Returns:
        The error's norm relative to the reference's.

    Raises:
        RuntimeError: If the testbench does not report a round halted.
    """
    entry = compiled.entry
    columns = entry["columns"]
    weights = pack_weights(weights=right).tobytes()
    extra = [pack_bias(bias=bias).tobytes()] if bias is not None else []
    drain_words = entry["drain_words"]
    product = np.empty((left.shape[0], columns), dtype=np.float32)
    for block in range(0, left.shape[0], _BLOCK_TOKENS):
        activations = pack_activations(activations=left[block:block + _BLOCK_TOKENS]).tobytes()
        dumped = run_image(compiled.image, arguments=[activations, weights, bytes(2 * drain_words), *extra],
                           dump_bytes=2 * drain_words, what=f"token block {block // _BLOCK_TOKENS} of {compiled.name}",
                           build_dir=build_dir, max_cycles=max_cycles)
        tiles = from_bf16_words(words=np.frombuffer(dumped, dtype="<u2")[_ZERO_TILE_WORDS:])
        product[block:block + _BLOCK_TOKENS] = (tiles.reshape(columns // _TILE, _BLOCK_TOKENS, _TILE)
                                                .transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns))

    staged = left.astype(np.float64)
    if entry["activate"]:
        staged = _on_bf16_grid(values=(0.5 * staged * (1.0 + np.vectorize(math.erf)(staged / math.sqrt(2.0))))
                               .astype(np.float32)).astype(np.float64)
    expected = staged @ right.astype(np.float64) + (0.0 if bias is None else bias)
    width = right.shape[1]
    return float(np.linalg.norm(product[:, :width] - expected) / np.linalg.norm(expected))


def draw_operands(*, depth: int, columns: int, seed: int = 0) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Returns one token block of activations and the weights, on the BF16 grid, the weights scaled so every product
    stays near one whatever the depth."""
    rng = np.random.default_rng(seed)
    left = _on_bf16_grid(values=rng.standard_normal((_BLOCK_TOKENS, depth)).astype(np.float32))
    right = _on_bf16_grid(values=(rng.standard_normal((depth, columns)) / math.sqrt(depth)).astype(np.float32))
    return left, right


def main() -> int:
    """Runs one product against the RTL and reports its error.

    Returns:
        Zero when the product matched within the tolerance the BF16 accumulate chain allows.

    Raises:
        ValueError: If only one of the activations and the weights is named, or the image is no GEMM.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default=None,
                        help="a Triton kernel to compile and run, as a Python file and its name after a colon")
    parser.add_argument("--activations", default=None, help="npy file holding the kernel's left operand")
    parser.add_argument("--weights", default=None, help="npy file holding the kernel's right operand")
    parser.add_argument("--depth", type=int, default=192, help="the contraction, without --kernel")
    parser.add_argument("--columns", type=int, default=48, help="output columns, without --kernel")
    parser.add_argument("--activate", action="store_true", help="pass the activations through the GELU")
    parser.add_argument("--bias", action="store_true", help="add a bias row in the same image")
    parser.add_argument("--build-dir", default=None, help="keep the Verilator build here across runs")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="cycles one round may take, past the bound the testbench carries")
    arguments = parser.parse_args()

    if (arguments.activations is None) != (arguments.weights is None):
        message = "Unable to read one operand alone. Name --activations and --weights together, or neither."
        raise ValueError(message)
    if arguments.kernel is not None:
        compiled = compile_kernel(load_kernel(arguments.kernel))
    else:
        compiled = compile_kernel(gemm_kernel(depth=arguments.depth, columns=arguments.columns,
                                              activate=arguments.activate, bias=arguments.bias))
    if compiled.entry["kind"] not in ("gemm", "gemm_bias"):
        message = f"Unable to run {compiled.name} as a GEMM. Its tile op is a {compiled.entry['kind']}."
        raise ValueError(message)

    entry = compiled.entry
    if arguments.activations is not None:
        left, right = np.load(arguments.activations), np.load(arguments.weights)
    else:
        left, right = draw_operands(depth=entry["depth"], columns=entry["columns"])
    bias = None
    if entry["kind"] == "gemm_bias":
        bias = _on_bf16_grid(values=np.random.default_rng(1).standard_normal(right.shape[1]).astype(np.float32))
    error = simulate(compiled, left=left, right=right, bias=bias, build_dir=arguments.build_dir,
                     max_cycles=arguments.max_cycles)

    verdict = "PASS" if error <= (_DEEP_TOLERANCE if left.shape[1] > _DEEP else _TOLERANCE) else "FAIL"
    print(f"{compiled.name}  {left.shape[0]}x{left.shape[1]} @ {right.shape[0]}x{right.shape[1]}, "
          f"{len(compiled.image) // 16} bundles  {verdict}  relative error {error * 100:.4f}%")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
