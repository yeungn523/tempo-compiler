"""Provides the compilation of one kernel into the image a board launches and what the runtime launches it with."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

from triton.runtime import JITFunction

from .backend import TempoBackend
from .frontend import make_ir

_TEXT_STAGES: tuple[str, ...] = ("tempo", "tempoisa", "bin")
"""Stages a kernel given as MLIR text runs: a tt module the "ttir" stage already left, or a kernel of tile ops, which
the conversion passes through unchanged."""


@dataclass(frozen=True, slots=True)
class Compiled:
    """Holds one compiled image and what the board runtime launches it with.

    Attributes:
        name: The image's file name, which the kernel's tile op fixes.
        image: The image bytes.
        entry: The image's manifest entry.
        constants: The constants block the runtime uploads beside the image unchanged, or None for an image reading
            none.
    """

    name: str
    image: bytes
    entry: dict
    constants: bytes | None

    def write(self, directory: pathlib.Path) -> pathlib.Path:
        """Writes the image, and its constants block under the name its entry states, into `directory`.

        Returns:
            The path the image was written to.
        """
        path = directory / self.name
        path.write_bytes(self.image)
        if self.constants is not None:
            (directory / self.entry["constants"]).write_bytes(self.constants)
        return path


def compile_kernel(kernel: JITFunction | str, *, heads: int = 1, **bound: int | float | bool) -> Compiled:
    """Compiles one kernel through every stage the backend registers.

    Args:
        kernel: The function the triton.jit decorator returned, or a kernel's MLIR text, a tt module or a tempo
            kernel of tile ops.
        heads: The heads an attention image walks, its Triton kernel attending one of them.
        **bound: The constexprs bound past a Triton kernel's defaults.

    Returns:
        The compiled image.

    Raises:
        ProgramError: If a stage refuses the kernel, carrying the MLIR pipeline's diagnostics.
    """
    # Triton types the target argument and then ignores it, so this backend carries no target descriptor.
    backend = TempoBackend(None)
    options = backend.parse_options({"attention_heads": heads})
    stages: dict = {}
    backend.add_stages(stages, options)

    if isinstance(kernel, str):
        source, names = kernel, _TEXT_STAGES
    else:
        source = make_ir(kernel, backend=backend, options=options, **bound)
        names = tuple(stages)
    metadata: dict = {}
    for name in names:
        source = stages[name](source, metadata)
    ((name, entry),) = metadata["manifest"].items()
    return Compiled(name=name, image=source, entry=entry, constants=metadata["constants"])
