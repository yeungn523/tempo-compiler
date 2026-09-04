"""Provides the Triton backend that compiles kernels for this machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from bf16 import _TILE
from triton.backends.compiler import BaseBackend

_OPERAND_ELEMENT: str = "bf16"
"""The only element type the systolic array supports."""

_MINIMUM_DOT_SHAPE: tuple[int, int, int] = (_TILE, _TILE, _TILE)
"""Smallest shape a dot may have, as rows by columns by depth, being the size of the systolic array itself."""


@dataclass(frozen=True, slots=True)
class _MinitpuOptions:
    """Holds the two compilation settings Triton's frontend reads.

    Neither is a choice this machine offers, so both are fixed and the backend ignores what a caller passes.

    Attributes:
        sanitize_overflow: Determines whether the frontend guards integer arithmetic against overflow. Setting it
            widens every integer binary operation narrower than 64 bits with an arith.extsi, recomputes it at 64 bits,
            compares the result against the narrow type's bounds with a pair of arith.cmpi joined by an arith.andi,
            and ends in a device assertion. A GEMM grows from 83 raw operations to 195. This machine executes no
            device assertion, so the guard costs the operations and reports nothing.
        allowed_dot_input_precisions: Precisions a kernel may name for its contraction. The array multiplies one
            element type at one precision, so the tensor float modes a GPU offers have no entry.
    """

    sanitize_overflow: bool = False
    allowed_dot_input_precisions: tuple[str, ...] = ("ieee",)


_OPTIONS: _MinitpuOptions = _MinitpuOptions()
"""Settings every compilation for this machine runs under."""


class MinitpuBackend(BaseBackend):
    """Compiles Triton kernels for this machine, stopping at the tt dialect.

    Triton declares six abstract methods. Four are empty because this machine adds no dialects, supplies no device
    specific modules, and runs no stage past the tt dialect, and a subclass omitting them cannot be instantiated.
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

    def parse_options(self, options: dict) -> _MinitpuOptions:
        """Returns the settings every compilation runs under, which a caller cannot override."""
        return _OPTIONS

    def get_codegen_implementation(self, options: _MinitpuOptions) -> dict:
        """Returns the dot callback, which is the one implementation the frontend requires."""
        return {"min_dot_size": _dot_shape}

    def load_dialects(self, context) -> None:
        """Loads the dialects this backend adds, of which there are none."""

    def get_module_map(self) -> dict[str, ModuleType]:
        """Returns the device specific implementations of Triton's interface modules, of which there are none."""
        return {}

    def add_stages(self, stages: dict, options: _MinitpuOptions) -> None:
        """Registers the compilation stages, of which this backend runs none past the tt dialect."""


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
