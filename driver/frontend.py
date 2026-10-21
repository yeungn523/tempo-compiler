"""Provides the Triton frontend, lowering a decorated kernel to the module Triton's make_ir builds."""

from __future__ import annotations

from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource
from triton.runtime import JITFunction

from .backend import MinitpuBackend, MinitpuOptions


def signature_of(kernel: JITFunction) -> dict[str, str]:
    """Returns the signature a kernel is compiled against, in the order the kernel declares its parameters.

    Args:
        kernel: The function the triton.jit decorator returned.

    Returns:
        Each parameter's Triton type string, read from its annotation, "constexpr" for a constexpr parameter and
        "i32" for an unannotated one.
    """
    return {parameter.name: "constexpr" if parameter.is_constexpr else _annotated(parameter.annotation) or "i32"
            for parameter in kernel.params}


def make_ir(kernel: JITFunction, *, backend: MinitpuBackend, options: MinitpuOptions,
            **bound: int | float | bool) -> ir.module:
    """Builds the module of one Triton kernel at the shape its constexprs fix, before any pass runs over it.

    Args:
        kernel: The function the triton.jit decorator returned.
        backend: The backend the module is built for.
        options: The settings the frontend reads.
        **bound: The constexprs bound past the kernel's defaults.

    Returns:
        The module.
    """
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=kernel, signature=signature_of(kernel), constexprs=declared | bound)
    return source.make_ir(None, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                          context)


def triton_kernel(kernel: JITFunction, **bound: int | float | bool) -> str:
    """Returns the tt module of one Triton kernel at the shape its constexprs fix, as the "ttir" stage leaves it.

    Args:
        kernel: The function the triton.jit decorator returned.
        **bound: The constexprs bound past the kernel's defaults.

    Returns:
        The module's MLIR text.
    """
    # Triton types the target argument and then ignores it, so this backend carries no target descriptor.
    backend = MinitpuBackend(None)
    options = backend.parse_options({})
    stages: dict = {}
    backend.add_stages(stages, options)
    return stages["ttir"](make_ir(kernel, backend=backend, options=options, **bound), {})


def _annotated(annotation: str) -> str:
    """Returns the Triton type one parameter's annotation states, unquoted.

    Args:
        annotation: The annotation the kernel declared, which `from __future__ import annotations` hands over as the
            source text and so with its quotes still on.

    Returns:
        The type, or the empty string for a parameter carrying no annotation.
    """
    text = annotation.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
