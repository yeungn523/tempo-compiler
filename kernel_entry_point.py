#!/usr/bin/env python3
"""Compiles one Triton kernel into the image and packed operands a board launches it from."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from compiler import ProgramError, compile_launch, plan, read_program, signature_of, stage  # noqa: E402
from compiler.compile import FusedLaunch  # noqa: E402
from asm import BUNDLE_BYTES, IRAM_DEPTH  # noqa: E402
from bf16 import (  # noqa: E402
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _ELEMENT_BYTES,
    _TILE,
    fused_ffn_reference,
    pack_activations,
    pack_fused_ffn,
    pack_weights,
)
from triton.runtime import JITFunction  # noqa: E402


def read_kernel(specification: str) -> JITFunction:
    """Returns the decorated kernel one file declares, loading that file as a module of its own.

    Args:
        specification: The kernel to compile, as the path to a Python file and, after a colon, the name it is
            declared under. A file declaring one kernel needs no name.

    Returns:
        The function the triton.jit decorator returned.

    Raises:
        FileNotFoundError: If the path names no file.
        ProgramError: If the file declares no decorated kernel, if it declares several and the call named none, or
            if the name it did give is not one.
    """
    location, _, name = specification.partition(":")
    path = pathlib.Path(location).expanduser().resolve()
    if not path.is_file():
        message = (
            f"Unable to read a kernel from {path}, which names no file. Pass the Python file the kernel is "
            f"declared in, naming the kernel after a colon when the file declares more than one."
        )
        raise FileNotFoundError(message)

    declared = {attribute: value for attribute, value in vars(_load_module(path)).items()
                if isinstance(value, JITFunction)}
    if not declared:
        message = (
            f"Unable to find a decorated kernel in {path}. A kernel this compiler reads is a function the "
            f"triton.jit decorator returned, and that file declares none."
        )
        raise ProgramError(message)
    if not name:
        if len(declared) > 1:
            message = (
                f"Unable to choose between the {len(declared)} kernels {path} declares, which are "
                f"{', '.join(sorted(declared))}. Name the one to compile after a colon."
            )
            raise ProgramError(message)
        return next(iter(declared.values()))
    if name not in declared:
        message = f"Unable to find a kernel named {name} in {path}, which declares {', '.join(sorted(declared))}."
        raise ProgramError(message)
    return declared[name]


def _pack_operands(*, path: pathlib.Path, left: NDArray[np.float32], right: NDArray[np.float32],
                   matmul, launch, staging, drain) -> None:
    """Writes the packed operands and every size a board allocates and reshapes from.

    Notes:
        The operands are packed into the traversal order the image walks, which the emitter settled laying the
        image out, so a board reads them where they lie. Packing runs here rather than on the board because the
        board is a slow core, and because these are the layouts `bf16` states for the image this call emitted.

    Args:
        path: The npz file to write.
        left: The activation operand at the extent the caller holds.
        right: The weight operand at the extent the caller holds.
        matmul: The matmul the kernel computes.
        launch: The planned launch the image was emitted for.
        staging: What the emitter says a host stages for that image.
        drain: The drain the plan states.
    """
    left, right = stage(left, right, matmul=matmul, launch=launch)
    activations = np.stack([pack_activations(activations=left[token : token + staging.rows])
                            for token in range(0, left.shape[0], staging.rows)])
    # Packs one weight buffer per column block, which is one whole buffer where the image walks the blocks
    # itself and one apiece where the host issues them.
    weights = np.stack([pack_weights(weights=right[:, column : column + staging.columns])
                        for column in range(0, right.shape[1], staging.columns)])
    np.savez(
        path,
        activations=activations,
        weights=weights,
        issue_order=",".join(axis for axis in launch.nest if axis != "k"),
        block_columns=staging.columns,
        expected=(left.astype(np.float64) @ right.astype(np.float64)).astype(np.float32),
        # The slack on each operand is the block the image prefetches past the last one.
        activation_bytes=activations[0].nbytes + staging.activation_slack,
        weight_bytes=weights[0].nbytes + _BLOCK_WEIGHT_BYTES,
        drain_words=drain.words,
        drain_bytes=drain.words * _ELEMENT_BYTES,
        zero_tile_words=drain.zero_tile_words,
        drain_shape=drain.shape,
    )


def _pack_fused_operands(*, path: pathlib.Path, activations: NDArray[np.float32],
                         up_weight: NDArray[np.float32], down_weight: NDArray[np.float32],
                         drain) -> None:
    """Writes the fused layer's packed operands and every size a board allocates and reshapes from.

    The layer reads three operands rather than a pair, and both projections ride one staging so that the two
    walk on one scalar register.

    Args:
        path: The npz file to write.
        activations: The layer input, whose rows are tokens.
        up_weight: The up projection, contracted against the input's columns.
        down_weight: The down projection, contracted against the hidden width.
        drain: The drain the plan states.
    """
    # One staged buffer per token block, the weights being the same for every one of them.
    blocks = [pack_fused_ffn(activations[token:token + _BLOCK_TOKENS], up_weight, down_weight)
              for token in range(0, activations.shape[0], _BLOCK_TOKENS)]
    staged, weights = np.stack([block[0] for block in blocks]), blocks[0][1]
    np.savez(
        path,
        activations=staged,
        weights=weights,
        expected=fused_ffn_reference(activations, up_weight, down_weight).astype(np.float32),
        activation_bytes=staged[0].nbytes,
        weight_bytes=weights.nbytes,
        drain_words=drain.words,
        drain_bytes=drain.words * _ELEMENT_BYTES,
        zero_tile_words=drain.zero_tile_words,
        drain_shape=drain.shape,
        output_columns=down_weight.shape[1],
    )


def _load_module(path: pathlib.Path) -> ModuleType:
    """Returns the module one Python file holds, executed under the file's own name.

    Args:
        path: The file to load.

    Returns:
        The executed module.

    Raises:
        ProgramError: If the interpreter cannot load the file as a module.
    """
    specification = importlib.util.spec_from_file_location(path.stem, path)
    if specification is None or specification.loader is None:
        message = (
            f"Unable to load {path} as a Python module. The file has to be one the interpreter imports, which a "
            f"file that is not Python source is not."
        )
        raise ProgramError(message)
    module = importlib.util.module_from_spec(specification)
    # Registers the module before running it, as the import system does, so a kernel declared in it is reached
    # through sys.modules by anything that looks its definition up by module name.
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _bindings(argument: str, *, flag: str) -> dict[str, str]:
    """Returns the name and value of each binding one flag carried.

    Args:
        argument: The value the flag was given, as names and values around equals signs, separated by commas.
        flag: The flag the value came from, for the message a malformed one raises.

    Returns:
        The bindings, keyed by name.

    Raises:
        ProgramError: If a binding carries no equals sign.
    """
    bindings = {}
    for binding in argument.split(",") if argument else ():
        name, separator, value = binding.strip().partition("=")
        if not separator or not name:
            message = f"Unable to read {binding} as a {flag} binding. Each one is a name and a value around an =."
            raise ProgramError(message)
        bindings[name] = value
    return bindings


def _integer_bindings(argument: str, *, flag: str) -> dict[str, int]:
    """Returns the name and integer value of each binding one flag carried.

    Args:
        argument: The value the flag was given, as names and integers around equals signs, separated by commas.
        flag: The flag the value came from, for the message a malformed one raises.

    Returns:
        The bindings, keyed by name.

    Raises:
        ProgramError: If a binding's value is not an integer.
    """
    bindings = {}
    for name, value in _bindings(argument, flag=flag).items():
        if not value.lstrip("-").isdigit():
            message = f"Unable to read {name}={value} as a {flag} binding, whose value has to be an integer."
            raise ProgramError(message)
        bindings[name] = int(value)
    return bindings


def _matmuls_covered(matmuls, launches) -> list[tuple[int, ...]]:
    """Returns the matmuls each launch runs, which is a pair for one image running both projections of a layer.

    Args:
        matmuls: The matmuls the kernel states, in program order.
        launches: The launches the plan made of them.

    Returns:
        One tuple of matmul positions per launch, in the order the host issues them.
    """
    fused = {position: (launch.produces, launch.consumes)
             for position, launch in enumerate(launches) if isinstance(launch, FusedLaunch)}
    paired = {position for pair in fused.values() for position in pair}
    # The planner appends a plain launch per matmul it did not fuse, in program order, and rewrites the
    # producer's slot for a pair, so the ones left over fill the remaining slots in that same order.
    alone = [position for position in range(len(matmuls)) if position not in paired]
    return [fused[position] if position in fused else (alone.pop(0),) for position in range(len(launches))]


def main() -> int:
    """Compiles the kernel one call names and prints the matmuls read from it and the launches planned for them.

    Returns:
        Zero once the plan is printed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", required=True,
                        help="the kernel to compile, as a Python file and the name it is declared under after a colon")
    parser.add_argument("--operands", default="",
                        help="Triton type of each parameter a buffer is passed for, as ap=*bf16,op=*fp32")
    parser.add_argument("--constexprs", default="",
                        help="value of each constexpr parameter, standing for the default the kernel declares")
    parser.add_argument("--strides", default="", help="value of each stride parameter at the launch")
    parser.add_argument("--dims", default="",
                        help="extent of each problem axis the module leaves open, keyed m, n or k")
    parser.add_argument("--output", default=None,
                        help="directory each launch's image is written to, or omit to plan without emitting")
    parser.add_argument("--activations", default=None,
                        help="npy file holding the left operand, which is packed beside the image it feeds")
    parser.add_argument("--weights", default=None,
                        help="npy file holding the right operand, named together with --activations")
    parser.add_argument("--up-weight", default=None,
                        help="npy file holding a fused layer's up projection, named with --down-weight")
    parser.add_argument("--down-weight", default=None,
                        help="npy file holding a fused layer's down projection, named with --up-weight")
    arguments = parser.parse_args()
    if (arguments.up_weight is None) != (arguments.down_weight is None):
        message = (
            "Unable to pack one projection alone. A fused layer contracts its input against the up projection "
            "and that product against the down projection, so name both or neither."
        )
        raise ProgramError(message)
    if arguments.weights is not None and arguments.up_weight is not None:
        message = (
            "Unable to read a weight operand two ways. --weights names the pair a plain product reads, and "
            "--up-weight with --down-weight the three a fused layer reads, so name one form or the other."
        )
        raise ProgramError(message)
    if (arguments.activations is not None) != (arguments.weights is not None
                                              or arguments.up_weight is not None):
        message = (
            "Unable to pack one operand alone. The operands are padded to the extent one launch covers and "
            "packed against each other, so name the activations and their weights together or neither."
        )
        raise ProgramError(message)
    if arguments.activations is not None and arguments.output is None:
        message = "Unable to pack operands without --output, which names the directory they are written to."
        raise ProgramError(message)

    kernel = read_kernel(arguments.kernel)
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel,
                           signature=signature_of(kernel, operands=_bindings(arguments.operands, flag="--operands")),
                           constexprs=declared | _integer_bindings(arguments.constexprs, flag="--constexprs"))
    # States the same extents for every matmul, since one call site holds one problem shape.
    dims = _integer_bindings(arguments.dims, flag="--dims")
    planned = plan(matmuls, strides=_integer_bindings(arguments.strides, flag="--strides"),
                   dims=(dims,) * len(matmuls))
    if arguments.activations is not None and planned.repack:
        rewritten = ", ".join(f"matmul {repacked.matmul}'s {repacked.role} steps by {repacked.stride} along "
                              f"{repacked.axis}" for repacked in planned.repack)
        message = (
            f"Unable to pack an operand this plan repacks, being {rewritten}. The packing below lays each "
            f"operand out where it already lies, so one whose lane axis steps by more than one would be packed "
            f"as though it were contiguous, and the reference written beside it would carry the same mistake. "
            f"Lay the operand out with a unit stride along that axis, or omit --activations to plan alone."
        )
        raise ProgramError(message)

    written = pathlib.Path(arguments.output).expanduser().resolve() if arguments.output else None
    if written is not None:
        written.mkdir(parents=True, exist_ok=True)

    covered = _matmuls_covered(matmuls, planned.launches)
    for index, launch in enumerate(planned.launches):
        for position in covered[index]:
            matmul = matmuls[position]
            print(f"matmul {position}  {matmul.rows}x{matmul.columns} tile over {matmul.depth}, reduction "
                  f"{matmul.reduction}, accumulator {matmul.accumulator}")
        issued = f"{launch.launches} launches" if launch.launches > 1 else "one launch"
        if isinstance(launch, FusedLaunch):
            print(f"fused  {index}  matmuls {launch.produces} and {launch.consumes} in one image, issued over "
                  f"{issued} of {_BLOCK_TOKENS} by {launch.output_tiles * _TILE}")
            print(f"launch {index}  row_blocks={launch.row_blocks} chunks={launch.chunks} "
                  f"steps={launch.steps} output_tiles={launch.output_tiles}")
        else:
            # The block covers whatever the nest keeps live inside its reduction, so every order that plans is
            # the order the launches walk.
            print(f"nest   {index}  kernel walks {', '.join(launch.nest)}, issued as written over "
                  f"{issued} of {launch.block_m} by {launch.block_n}")
            print(f"launch {index}  row_blocks={launch.row_blocks} column_tiles={launch.column_tiles} "
                  f"groups={launch.groups} loop_blocks={launch.loop_blocks} activate={launch.activate} "
                  f"fuse_groups={launch.fuse_groups}")
        try:
            image, staging = compile_launch(launch)
        except NotImplementedError as error:
            print(f"image  {index}  not emitted: {error}")
            return 1
        drain = planned.drains[index]
        bundles = len(image) // BUNDLE_BYTES
        print(f"image  {index}  {bundles} bundles over {staging.rows}x{staging.columns}, "
              f"{100 * bundles / IRAM_DEPTH:.0f}% of IRAM, draining {drain.words} words shaped {drain.shape}")
        if written is not None:
            target = written / f"matmul_{index}.bin"
            target.write_bytes(image)
            print(f"wrote  {index}  {target}")
        if written is not None and arguments.activations is not None:
            packed = written / f"matmul_{index}_operands.npz"
            if isinstance(launch, FusedLaunch):
                if arguments.up_weight is None:
                    message = (
                        f"Unable to pack the operands of launch {index}, which runs both projections of a "
                        f"feed-forward layer and so reads three of them rather than a pair. Name them with "
                        f"--up-weight and --down-weight, or omit --activations to compile the image alone."
                    )
                    raise ProgramError(message)
                _pack_fused_operands(path=packed, activations=np.load(arguments.activations),
                                     up_weight=np.load(arguments.up_weight),
                                     down_weight=np.load(arguments.down_weight), drain=drain)
            else:
                _pack_operands(path=packed, left=np.load(arguments.activations),
                               right=np.load(arguments.weights),
                               matmul=matmuls[covered[index][0]], launch=launch, staging=staging, drain=drain)
            print(f"packed {index}  {packed}")
    for padded in planned.padding:
        print(f"padding   matmul {padded.matmul} covers {padded.extent} of {padded.axis} by {padded.padded}")
    for repacked in planned.repack:
        print(f"repack    matmul {repacked.matmul} {repacked.role} along {repacked.axis}, which "
              f"{repacked.base} steps by {repacked.stride}")
    print(f"plan      {planned.issued_over_useful:.3f} multiplies issued per multiply the problems hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
