#!/usr/bin/env python3
"""Writes the tt dialect module of every kernel a GPT-2 small layer launches, at the shapes the layer compiles it at.

The modules are the input the MLIR pipeline under mlir/ reads. They are written in the form Triton prints, without
locations, which minitpu-opt parses because it is built from the LLVM commit and dialect sources the Triton 3.8.0
wheel was.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from codegen import signature_of  # noqa: E402
from codegen.frontend import compile_to_ttir  # noqa: E402
from gpt2_triton import (gpt2_add, gpt2_add_bias, gpt2_attention, gpt2_decode_attention,  # noqa: E402
                         gpt2_layernorm, gpt2_matmul, gpt2_matmul_gelu)
from triton.runtime import JITFunction  # noqa: E402

_MODEL: int = 768
"""GPT-2 small's model width."""

_KERNELS: tuple[tuple[str, JITFunction, dict[str, int]], ...] = (
    ("gemm_768x2304", gpt2_matmul,
     {"N": 3 * _MODEL, "K": _MODEL, "stride_am": _MODEL, "stride_bk": 3 * _MODEL, "stride_cm": 3 * _MODEL}),
    ("gemm_3072x768_gelu", gpt2_matmul_gelu,
     {"N": _MODEL, "K": 4 * _MODEL, "stride_am": 4 * _MODEL, "stride_bk": _MODEL, "stride_cm": _MODEL}),
    ("add_768", gpt2_add, {"N": _MODEL}),
    ("add_bias_768", gpt2_add_bias, {"N": _MODEL}),
    ("layernorm_768", gpt2_layernorm, {"N": _MODEL}),
    ("attention_64", gpt2_attention, {"M": 64}),
    ("decode_attention_256", gpt2_decode_attention, {"N": 256, "LENGTH": 256}),
)
"""Name each module is written under, the kernel, and the constexprs the layer binds past the kernel's defaults,
matching tools/build_gpt2_images.py."""


def main() -> int:
    """Writes one module per kernel the table above lists.

    Returns:
        Zero once every module is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_HERE / "mlir" / "test" / "Inputs"),
                        help="Directory the modules are written into.")
    arguments = parser.parse_args()

    written = pathlib.Path(arguments.output).expanduser().resolve()
    written.mkdir(parents=True, exist_ok=True)
    for name, kernel, bound in _KERNELS:
        declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
        module = compile_to_ttir(kernel, signature=signature_of(kernel, operands={}), constexprs=declared | bound)
        path = written / f"{name}.ttir"
        # Ends the file in the newline MLIR's own printer writes and Triton's binding leaves off.
        path.write_text(module.str_nodebug().rstrip("\n") + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
