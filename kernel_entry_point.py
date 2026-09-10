#!/usr/bin/env python3
"""Passes one Triton kernel to the compiler and reports the launches it plans."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from types import ModuleType

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))

from compiler import ProgramError, plan, read_program, signature_of  # noqa: E402
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
    arguments = parser.parse_args()

    kernel = read_kernel(arguments.kernel)
    declared = {parameter.name: parameter.default for parameter in kernel.params if parameter.is_constexpr}
    matmuls = read_program(kernel,
                           signature=signature_of(kernel, operands=_bindings(arguments.operands, flag="--operands")),
                           constexprs=declared | _integer_bindings(arguments.constexprs, flag="--constexprs"))
    # States the same extents for every matmul, since one call site holds one problem shape.
    dims = _integer_bindings(arguments.dims, flag="--dims")
    planned = plan(matmuls, strides=_integer_bindings(arguments.strides, flag="--strides"),
                   dims=(dims,) * len(matmuls))

    for index, (matmul, launch) in enumerate(zip(matmuls, planned.launches, strict=True)):
        print(f"matmul {index}  {matmul.rows}x{matmul.columns} tile over {matmul.depth}, reduction "
              f"{matmul.reduction}, accumulator {matmul.accumulator}")
        print(f"launch {index}  row_blocks={launch.row_blocks} column_tiles={launch.column_tiles} "
              f"groups={launch.groups} loop_blocks={launch.loop_blocks} activate={launch.activate} "
              f"fuse_groups={launch.fuse_groups}")
    for padded in planned.padding:
        print(f"padding   matmul {padded.matmul} covers {padded.extent} of {padded.axis} by {padded.padded}")
    for repacked in planned.repack:
        print(f"repack    matmul {repacked.matmul} {repacked.role} along {repacked.axis}, which "
              f"{repacked.base} steps by {repacked.stride}")
    print(f"plan      {planned.issued_over_useful:.3f} multiplies issued per multiply the problems hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
