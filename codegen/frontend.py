"""Provides the Triton frontend, lowering a decorated kernel to the tt dialect."""

from __future__ import annotations

from collections.abc import Callable

from triton._C.libtriton import ir, passes
from triton.compiler.compiler import ASTSource
from triton.runtime import JITFunction

from .backend import MinitpuBackend

_PIPELINE: tuple[Callable[[ir.pass_manager], None], ...] = (
    passes.common.add_inliner,
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer,
    passes.common.add_canonicalizer,
    passes.ttir.add_combine,
    passes.ttir.add_reorder_broadcast,
    passes.common.add_cse,
    passes.common.add_symbol_dce,
)
"""Passes run over the tt dialect. Loop unrolling is omitted because a kernel naming an unroll factor would lose the
contraction loop the address generator levels bind to."""


def compile_to_ttir(kernel: JITFunction, *, signature: dict[str, str], constexprs: dict[str, int],
                    canonicalize: bool = True) -> ir.module:
    """Lowers a decorated Triton kernel to the tt dialect.

    Args:
        kernel: The function the triton.jit decorator returned.
        signature: Parameter name to Triton type string, in declaration order, marking folded parameters "constexpr".
        constexprs: Values of the constexpr parameters, which fix the tile.
        canonicalize: Determines whether the pass pipeline runs over the module. Clearing it leaves the overflow
            scaffolding and the uninlined helpers Triton's own frontend emits.

    Returns:
        The lowered module.
    """
    # Triton types the target argument and then ignores it, so this backend carries no target descriptor.
    backend = MinitpuBackend(None)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    module = source.make_ir(None, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    if not canonicalize:
        return module

    manager = ir.pass_manager(module.context)
    for add_pass in _PIPELINE:
        add_pass(manager)
    manager.run(module, "make_ttir")
    return module
