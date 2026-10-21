"""Provides the Triton backend that compiles kernels for this machine, its stages past the tt dialect running the MLIR
pipeline under mlir/.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
import subprocess
from types import ModuleType

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend

TOOLS: pathlib.Path = pathlib.Path(__file__).resolve().parents[1] / "mlir" / "build" / "bin"
"""Stores the directory minitpu-opt and minitpu-translate are built into."""

_OPERAND_ELEMENT: str = "bf16"
"""The only element type the systolic array supports."""

_ARRAY: int = 16
"""Rows and columns of the systolic array."""

_MINIMUM_DOT_SHAPE: tuple[int, int, int] = (_ARRAY, _ARRAY, _ARRAY)
"""Smallest shape a dot may have, as rows by columns by depth, being the size of the systolic array itself."""

_TTIR_PASSES = (
    passes.common.add_inliner,
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer,
    passes.common.add_canonicalizer,
    passes.ttir.add_combine,
    passes.ttir.add_reorder_broadcast,
    passes.common.add_cse,
    passes.common.add_symbol_dce,
)
"""Passes run over the tt dialect. Loop unrolling is omitted because the conversion reads each loop by what it walks,
which an unrolled loop no longer states."""

_LOWERING: str = (
    "builtin.module(minitpu.kernel(minitpu-tile-to-vector,minitpu-allocate-vmem,minitpu-allocate-registers),"
    "convert-minitpu-to-isa,minitpuisa.program(minitpuisa-pack,minitpuisa-schedule))"
)
"""Passes taking a kernel of tile ops to a scheduled minitpuisa program."""


class ProgramError(RuntimeError):
    """Raised when the MLIR pipeline refuses a kernel, carrying the diagnostics it printed."""


@dataclass(frozen=True, slots=True)
class MinitpuOptions:
    """Holds the compilation settings Triton's frontend reads and the one this backend's own stages read.

    Attributes:
        sanitize_overflow: Determines whether the frontend guards integer arithmetic against overflow. Setting it
            widens every integer binary operation narrower than 64 bits with an arith.extsi, recomputes it at 64 bits,
            compares the result against the narrow type's bounds with a pair of arith.cmpi joined by an arith.andi,
            and ends in a device assertion. This machine executes no device assertion, so the guard costs the
            operations and reports nothing. Fixed off.
        allowed_dot_input_precisions: Precisions a kernel may name for its contraction. The array multiplies one
            element type at one precision, so the tensor float modes a GPU offers have no entry. Fixed.
        attention_heads: Heads an attention image walks, its Triton kernel attending one of them. Every other kernel
            ignores it.
    """

    sanitize_overflow: bool = False
    allowed_dot_input_precisions: tuple[str, ...] = ("ieee",)
    attention_heads: int = 1


class MinitpuBackend(BaseBackend):
    """Compiles Triton kernels for this machine, from the tt dialect through minitpu tile ops to an image.

    Triton declares six abstract methods. Two are empty because this machine adds no dialects to Triton's context and
    supplies no device specific modules. `get_codegen_implementation` is a seventh that BaseBackend does not declare
    and every backend must still carry: `make_ir` takes its return value as an argument.
    """

    @staticmethod
    def supports_target(target) -> bool:
        """Returns True, because this backend is constructed directly rather than dispatched to by target.

        Triton dispatches on the target only when it selects a backend out of its registry. Registering this one there
        is what would make the answer depend on the argument.
        """
        return True

    def hash(self) -> str:
        """Returns the identifier distinguishing this backend's cache entries."""
        return f"minitpu-{_OPERAND_ELEMENT}-{_MINIMUM_DOT_SHAPE}"

    def parse_options(self, options: dict) -> MinitpuOptions:
        """Returns the settings a compilation runs under, of which a caller chooses only the attention heads."""
        return MinitpuOptions(attention_heads=options.get("attention_heads", 1))

    def get_codegen_implementation(self, options: MinitpuOptions) -> dict:
        """Returns the dot callback, which is the one implementation the frontend requires."""
        return {"min_dot_size": _dot_shape}

    def load_dialects(self, context) -> None:
        """Loads the dialects this backend adds to Triton's context, of which there are none, its own living in the
        MLIR tools."""

    def get_module_map(self) -> dict[str, ModuleType]:
        """Returns the device specific implementations of Triton's interface modules, of which there are none."""
        return {}

    def add_stages(self, stages: dict, options: MinitpuOptions, *_) -> None:
        """Registers the compilation stages, each taking the previous stage's output and the shared metadata.

        Notes:
            The stages are "ttir", canonicalizing the module make_ir builds into tt dialect text; "minitpu",
            converting it into a kernel of tile ops and recording its manifest entry and any constants block in the
            metadata; "minitpuisa", lowering the tile ops to a scheduled program; and "bin", writing the image.
            Absorbs the arguments past the two BaseBackend declares, because Triton's own driver passes a third:
            `compiler.py` calls `backend.add_stages(stages, options, src.language)`.
        """
        stages["ttir"] = lambda source, metadata: _make_ttir(source)
        stages["minitpu"] = lambda source, metadata: _make_minitpu(source, metadata, heads=options.attention_heads)
        stages["minitpuisa"] = lambda source, metadata: _run_tool("minitpu-opt", f"-pass-pipeline={_LOWERING}",
                                                                  source=source).decode()
        stages["bin"] = lambda source, metadata: _run_tool("minitpu-translate", "--export-minitpu-image",
                                                           source=source)


def _make_ttir(module: ir.module) -> str:
    """Runs Triton's tt passes over the module make_ir built and returns it as text."""
    manager = ir.pass_manager(module.context)
    for add_pass in _TTIR_PASSES:
        add_pass(manager)
    manager.run(module, "make_ttir")
    # Ends the text in the newline MLIR's own printer writes and Triton's binding leaves off.
    return module.str_nodebug().rstrip("\n") + "\n"


def _make_minitpu(source: str, metadata: dict, *, heads: int) -> str:
    """Converts a tt module, or a kernel already stated as tile ops, into a kernel of tile ops, recording its manifest
    entry under "manifest" and its constants block, when the image reads one unchanged, under "constants"."""
    kernel = _run_tool("minitpu-opt", f"--convert-tt-to-minitpu=attention-heads={heads}", source=source).decode()
    metadata["manifest"] = json.loads(_run_tool("minitpu-translate", "--export-minitpu-manifest", source=kernel))
    (entry,) = metadata["manifest"].values()
    metadata["constants"] = (_run_tool("minitpu-translate", "--export-minitpu-constants", source=kernel)
                             if "constants" in entry else None)
    return kernel


def _run_tool(tool: str, *arguments: str, source: str) -> bytes:
    """Runs one of the MLIR tools over MLIR text and returns what it writes.

    Args:
        tool: The tool's name in `TOOLS`.
        *arguments: The tool's arguments past its input and output.
        source: The MLIR text the tool reads.

    Returns:
        The tool's output.

    Raises:
        ProgramError: If the tool refuses its input, carrying its diagnostics.
    """
    run = subprocess.run([str(TOOLS / tool), "-", *arguments, "-o", "-"], input=source.encode(), capture_output=True)
    if run.returncode:
        message = f"Unable to compile the kernel. {tool} refused it:\n{run.stderr.decode()}"
        raise ProgramError(message)
    return run.stdout


def _dot_shape(lhs, rhs) -> tuple[int, int, int]:
    """Returns the smallest shape a dot may have, refusing operands the array cannot multiply.

    Args:
        lhs: Type of the left operand.
        rhs: Type of the right operand.

    Returns:
        The minimum shape, as rows by columns by depth.

    Raises:
        TypeError: If either operand is not the element type the array multiplies.
    """
    for side, operand in (("left", lhs), ("right", rhs)):
        if str(operand.scalar) != _OPERAND_ELEMENT:
            message = (
                f"Unable to contract a {side} operand of {operand.scalar}. It must be {_OPERAND_ELEMENT}, because "
                f"that is the only element type the systolic array multiplies. Convert the operand before the dot, "
                f"or stage it as {_OPERAND_ELEMENT} on the host."
            )
            raise TypeError(message)
    return _MINIMUM_DOT_SHAPE
