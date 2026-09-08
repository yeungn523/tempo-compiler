#!/usr/bin/env python3
"""Generates the published ISA operation table, the RTL conformance package and the assembler's scheduling model.

compiler/minitpuisa.td is the single source of truth for the implemented scheduling contract, and it is read through
llvm-tblgen --dump-json, so nothing here needs an MLIR build. The artifacts derived from it are the operation table
spliced between the generated markers in docs/ISA_AND_INTERFACES.md, the localparams in tb/isa_latency_pkg.sv and the
per-opcode facts in board_package/isa_model.py. Running this script with --write regenerates all three, and running it
with --check exits non-zero when any of them, or any scheduling constant in board_package/asm.py, disagrees with the
source.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any

_REPO: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""Stores the path to the repository root."""

_SOURCE: pathlib.Path = _REPO / "compiler" / "minitpuisa.td"
"""Stores the path to the TableGen contract every generated artifact derives from."""

_DOC: pathlib.Path = _REPO / "docs" / "ISA_AND_INTERFACES.md"
"""Stores the path to the markdown page carrying the generated operation table."""

_PACKAGE: pathlib.Path = _REPO / "tb" / "isa_latency_pkg.sv"
"""Stores the path to the package tb_isa_conformance.sv holds the sequencer's own localparams against."""

_MODEL: pathlib.Path = _REPO / "board_package" / "isa_model.py"
"""Stores the path to the module the assembler reads its per-opcode scheduling facts from."""

_ASSEMBLER: pathlib.Path = _REPO / "board_package" / "asm.py"
"""Stores the path to the assembler whose hand-written scheduling constants are checked against the source."""

_BEGIN: str = "<!-- BEGIN GENERATED: tools/gen_isa_doc.py -->"
"""Stores the marker opening the generated region of the markdown page."""

_END: str = "<!-- END GENERATED -->"
"""Stores the marker closing the generated region of the markdown page."""

_HEADER: str = "Run `python3 tools/gen_isa_doc.py --write` to generate."
"""Stores the line the generated SystemVerilog package and Python module open with."""

_COLUMNS: tuple[str, ...] = (
    "Physical slot / operation",
    "VREG reads",
    "VREG writes",
    "W",
    "Earliest consumer L",
    "Issue interval / structural requirement",
)
"""Stores the header cells of the generated operation table."""

_TBLGEN_CANDIDATES: tuple[str, ...] = (
    "llvm-tblgen",
    "llvm-tblgen-18",
    "llvm-tblgen-17",
    "llvm-tblgen-19",
    "llvm-tblgen-20",
)
"""Stores the binaries tried in order when LLVM_TBLGEN does not name one."""


def main() -> int:
    """Regenerates every artifact or reports the ones that disagree with the source.

    Returns:
        Zero when the run succeeds, and one when --check finds a stale artifact or a drifted assembler constant.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate every artifact")
    mode.add_argument("--check", action="store_true", help="exit non-zero if any artifact is stale")
    arguments = parser.parse_args()

    data = _load()
    artifacts = _artifacts(data)
    names = ", ".join(str(path.relative_to(_REPO)) for path in artifacts)

    if arguments.write:
        for path, content in artifacts.items():
            path.write_text(content)
        print(f"wrote {names}")
        return 0

    stale = [path.relative_to(_REPO) for path, content in artifacts.items()
             if not path.exists() or path.read_text() != content]
    if stale:
        for path in stale:
            print(f"STALE: {path} does not match {_SOURCE.relative_to(_REPO)}", file=sys.stderr)
        print("run: python3 tools/gen_isa_doc.py --write", file=sys.stderr)
        return 1

    drift = _check_assembler_constants(data)
    if drift:
        for message in drift:
            print(f"DRIFT: {message}", file=sys.stderr)
        return 1

    print(f"{names} are in sync with {_SOURCE.relative_to(_REPO)}")
    return 0


def _load() -> dict[str, Any]:
    """Returns the source's records reshaped into the mapping every renderer below reads."""
    records = _records()
    return {
        "profile": _profile(records),
        "operations": _operations(records),
        "rtl_params": {record["name"]: record["value"]
                       for record in _ordered(records, "RtlParameter", "rtl_parameters")},
        "matrix": _matrix(records),
        "resources": _resources(records),
    }


def _records() -> dict[str, Any]:
    """Returns the TableGen record dump of the source.

    Raises:
        SystemExit: If tblgen rejects the file, carrying its own diagnostic rather than a summary of one.
    """
    completed = subprocess.run(args=[_tblgen(), "--dump-json", str(_SOURCE)], capture_output=True, text=True)
    if completed.returncode:
        message = f"llvm-tblgen rejected {_SOURCE.relative_to(_REPO)}:\n{completed.stderr.strip()}"
        raise SystemExit(message)
    return json.loads(completed.stdout)


def _tblgen() -> str:
    """Returns the llvm-tblgen to read the source with.

    Raises:
        SystemExit: If none is on PATH, naming the environment variable that overrides the search.
    """
    override = os.environ.get("LLVM_TBLGEN")
    if override:
        return override

    for candidate in _TBLGEN_CANDIDATES:
        found = shutil.which(candidate)
        if found is not None:
            return found

    message = (
        f"Unable to read {_SOURCE.relative_to(_REPO)} without llvm-tblgen, which is not on PATH under any of "
        f"{', '.join(_TBLGEN_CANDIDATES)}. Install an LLVM package or set LLVM_TBLGEN to the binary."
    )
    raise SystemExit(message)


def _referenced(records: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolves a field holding a list of record references to the records themselves."""
    return [records[entry["def"]] for entry in entries]


def _ordered(records: dict[str, Any], record_class: str, field: str) -> list[dict[str, Any]]:
    """Returns every record of the named class, in the order IsaOrder lists them.

    TableGen sorts its own record listing by record name rather than by definition order, and the order the records
    publish in is part of the contract, so the source names that order outright.

    Raises:
        SystemExit: If IsaOrder omits a record. It is refused rather than dropped, because a mnemonic missing from
            the generated model is a hazard the scheduler cannot then see.
    """
    listed = _referenced(records, records["IsaOrder"][field])
    named = {record["!name"] for record in listed}
    missing = [name for name in records["!instanceof"].get(record_class, ()) if name not in named]
    if missing:
        plural = "s" if len(missing) > 1 else ""
        message = (
            f"{_SOURCE.relative_to(_REPO)} defines {record_class} record{plural} {', '.join(missing)} that "
            f"IsaOrder.{field} does not list, so {'they' if plural else 'it'} would not reach the "
            f"generated artifacts. Add {'them' if plural else 'it'} to the list."
        )
        raise SystemExit(message)
    return listed


def _profile(records: dict[str, Any]) -> dict[str, int]:
    """Returns the array geometry every published offset is quoted against.

    Notes:
        Every integer field of the record reaches the model, rather than the ones this function happens to name, so
        a capacity added to Profile is published without an edit here and cannot sit in the source unread while a
        consumer hand-copies it. The fields sort by name, because the order is otherwise tblgen's to change.
    """
    record = records["MiniTpuProfile"]
    return {name: value for name, value in sorted(record.items())
            if not name.startswith("!") and isinstance(value, int) and not isinstance(value, bool)}


def _operations(records: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns one entry per operation group, shaped as the renderers below read it.

    Notes:
        An empty field is omitted rather than emitted, because the renderers select on whether a group names opcodes
        at all and on whether it overrides its consumer offset.
    """
    groups = []
    for record in _ordered(records, "OpGroup", "operations"):
        entry = {
            "slot": record["slot"],
            "ops": record["mnemonics"],
            "vreg_reads": record["vreg_reads"],
            "vreg_writes": record["vreg_writes"],
            "w": _writeback(record),
            "issue": record["issue_requirement"],
            "reads": record["register_reads"],
            "writes": record["register_write"],
            "port_a": bool(record["claims_read_port_a"]),
        }
        optional = (
            ("opcodes", record["opcodes"]),
            ("rtl_enum", record["rtl_enumerators"]),
            ("rtl_param", record["rtl_parameter"]),
            ("consumer_override", record["consumer_override"]),
        )
        entry.update({key: value for key, value in optional if value})
        groups.append(entry)
    return groups


def _writeback(record: dict[str, Any]) -> int | str | None:
    """Returns the writeback offset as the artifacts carry it, being absent, one offset, or a range of beats.

    An uninitialised offset states that the group reserves no write port at all, which is a different claim from
    reserving it at cycle zero, so it stays distinguishable from a value here.
    """
    if record["writeback_offset"] is None:
        return None
    if record["writeback_last_offset"] is None:
        return record["writeback_offset"]
    return f"{record['writeback_offset']}..{record['writeback_last_offset']}"


def _matrix(records: dict[str, Any]) -> dict[str, Any]:
    """Returns the matrix commands' occupancy, result latency and port-A ownership."""
    commands = _ordered(records, "MatrixCommand", "matrix_commands")
    return {
        "occupancy": {command["mnemonic"]: command["occupancy_cycles"] for command in commands},
        "result_latency": {command["mnemonic"]: command["result_latency_cycles"] for command in commands
                           if command["result_latency_cycles"] is not None},
        "port_a_held_by": [command["mnemonic"] for command in commands if command["holds_read_port_a"]],
    }


def _resources(records: dict[str, Any]) -> dict[str, Any]:
    """Returns each capacity's values, RTL parameter, assembler constants and note."""
    resources = {}
    for record in _ordered(records, "Resource", "resources"):
        spec = {value["name"]: value["value"] for value in _referenced(records, record["values"])}
        if record["rtl_parameter"]:
            spec["rtl_param"] = record["rtl_parameter"]
        spec["asm_constants"] = {constant["name"]: constant["resource_field"]
                                 for constant in _referenced(records, record["asm_constants"])}
        if record["note"]:
            spec["note"] = record["note"]
        resources[record["group"]] = spec
    return resources


def _artifacts(data: dict[str, Any]) -> dict[pathlib.Path, str]:
    """Returns the content each generated artifact must hold, keyed by its path."""
    return {
        _DOC: _splice(_DOC.read_text(), _render_table(data)),
        _PACKAGE: _render_package(data),
        _MODEL: _render_model(data),
    }


def _splice(doc_text: str, table: str) -> str:
    """Replaces the generated region of the markdown page with the rendered table.

    Raises:
        SystemExit: If the page has lost either of the markers bounding that region.
    """
    if _BEGIN not in doc_text or _END not in doc_text:
        message = f"{_DOC} is missing the {_BEGIN} / {_END} markers around its table"
        raise SystemExit(message)
    head, rest = doc_text.split(_BEGIN, 1)
    _, tail = rest.split(_END, 1)
    return f"{head}{_BEGIN}\n{table}\n{_END}{tail}"


def _render_table(data: dict[str, Any]) -> str:
    """Returns the markdown region carrying the operation, matrix command and capacity tables."""
    rows = ["| " + " | ".join(_COLUMNS) + " |",
            "| --- | --- | --- | ---: | ---: | --- |"]
    for entry in data["operations"]:
        # Renders mnemonics as code and leaves prose descriptions plain.
        names = ", ".join(f"`{mnemonic}`" if mnemonic.replace("_", "").isalnum() else mnemonic
                          for mnemonic in entry["ops"])
        rows.append("| " + " | ".join([
            f"{entry['slot']}: {names}",
            entry["vreg_reads"],
            entry["vreg_writes"],
            "—" if entry["w"] is None else str(entry["w"]),
            _consumer(entry),
            entry["issue"],
        ]) + " |")

    matrix = data["matrix"]
    rows += ["", "### Matrix commands", "",
             "| Command | Controller occupancy | Result latency | Holds VREG read port A |",
             "| --- | ---: | ---: | :---: |"]
    for command, occupancy in matrix["occupancy"].items():
        latency = matrix["result_latency"].get(command)
        rows.append("| " + " | ".join([
            f"`{command}`",
            f"{occupancy} cycles",
            f"{latency} cycles" if latency is not None else "—",
            "yes" if command in matrix["port_a_held_by"] else "no",
        ]) + " |")
    rows += ["",
             "A command issued at *t* leaves the controller busy over [*t*+1, *t*+occupancy], so the "
             "next is legal at *t*+occupancy+1. Results leave the array in push order and one "
             "`vmatpop` drains exactly one `vmatpush`'s worth, so a pop waits for the **oldest** "
             "un-popped push. A pop placed earlier is still correct — `VPOP_STREAM` gates every beat "
             "on `mxu_output_valid_i` — but it stalls there holding the controller.", ""]

    rows += ["### Capacities a schedule can exceed", "",
             "| Resource | Value | RTL parameter | Consequence of exceeding it |",
             "| --- | ---: | --- | --- |"]
    for group, spec in data["resources"].items():
        capacity = _capacity(spec)
        rows.append("| " + " | ".join([
            group.replace("_", " "),
            str(capacity) if capacity is not None else "—",
            f"`{spec['rtl_param']}`" if "rtl_param" in spec else "—",
            spec.get("note", "—"),
        ]) + " |")
    fifo = data["resources"]["mxu_output_fifo"]
    rows += ["",
             f"One `vmatpush` produces {fifo['entries_per_vmatpush']} entries in each lane's output "
             f"FIFO and one `vmatpop` consumes {fifo['entries_per_vmatpop']}, so a burst is capped "
             f"at {fifo['depth'] // fifo['entries_per_vmatpush']} pushes before a pop must drain it.",
             ""]
    return "\n".join(rows)


def _consumer(entry: dict[str, Any]) -> str:
    """Returns the earliest legal consumer cycle, being the writeback offset plus one unless the entry overrides it."""
    if "consumer_override" in entry:
        return entry["consumer_override"]

    writeback = entry["w"]
    if writeback is None:
        return "—"
    if isinstance(writeback, str):
        # A range such as "3..6" reserves the write port through its last beat.
        return str(int(writeback.split("..")[1]) + 1)
    return str(writeback + 1)


def _capacity(spec: dict[str, Any]) -> int | None:
    """Returns the resource's headline value, being the one its first assembler constant names.

    A resource binding no assembler constant falls back to its first value, so a capacity no constant mirrors still
    publishes a number rather than a dash.
    """
    field = next(iter(spec["asm_constants"].values()), None)
    if field is not None:
        return spec[field]
    values = [name for name in spec if name not in {"rtl_param", "asm_constants", "note"}]
    return spec[values[0]] if values else None


def _render_package(data: dict[str, Any]) -> str:
    """Returns the generated SystemVerilog package carrying one localparam per published offset and opcode."""
    lines = [f"// {_HEADER}", "", "package isa_latency_pkg;"]

    params = dict(data["rtl_params"])
    for spec in data["resources"].values():
        capacity = _capacity(spec)
        if "rtl_param" in spec and capacity is not None:
            params[spec["rtl_param"]] = capacity
    for name, value in params.items():
        lines.append(f"  localparam int unsigned DOC_{name} = {value};")

    for entry in data["operations"]:
        for name, value in zip(entry.get("rtl_enum", ()), entry.get("opcodes", ())):
            lines.append(f"  localparam int unsigned DOC_{name} = {value};")

    lines += ["endpackage : isa_latency_pkg", ""]
    return "\n".join(lines)


def _render_model(data: dict[str, Any]) -> str:
    """Returns the generated Python module carrying the per-opcode scheduling facts the assembler reads."""
    lines = [
        '"""Provides the per-opcode scheduling facts the assembler schedules against.',
        "",
        _HEADER,
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "PROFILE: dict[str, int] = {",
    ]
    for name, value in data["profile"].items():
        lines.append(f"    {_python_literal(name)}: {value},")
    lines += [
        "}",
        '"""Array geometry every offset below is quoted against, and every capacity a consumer derives from it."""',
        "",
        "RTL_PARAMS: dict[str, int] = {",
    ]
    for name, value in data["rtl_params"].items():
        lines.append(f"    {_python_literal(name)}: {value},")
    lines += [
        "}",
        '"""Writeback reservation offset of each sequencer_pkg localparam tb_isa_conformance checks."""',
        "",
        "MATRIX_OCCUPANCY: dict[str, int] = {",
    ]
    for name, cycles in data["matrix"]["occupancy"].items():
        lines.append(f"    {_python_literal(name)}: {cycles},")
    lines += [
        "}",
        '"""Cycles each matrix command holds the controller, measured against the RTL."""',
        "",
        "MATRIX_PORT_A: tuple[str, ...] = " + repr(tuple(data["matrix"]["port_a_held_by"])).replace("'", '"'),
        '"""Matrix commands owning VREG read port A while they stream."""',
        "",
        f"MATRIX_RESULT_LATENCY: int = {data['matrix']['result_latency']['vmatpush']}",
        '"""Cycles from a vmatpush to its first result leaving the array."""',
        "",
        "OPERATIONS: tuple[dict, ...] = (",
    ]
    for entry in data["operations"]:
        # The X slot reaches the model without opcodes of its own, and every other group earns its entry by
        # naming at least one.
        if "opcodes" not in entry and entry["slot"] != "X":
            continue
        lines.append("    {")
        for key in ("slot", "ops", "opcodes", "reads", "writes", "w", "port_a"):
            value = entry.get(key, []) if key == "opcodes" else entry[key]
            lines.append(f"        {_python_literal(key)}: {_python_literal(value)},")
        lines.append("    },")
    lines += [
        ")",
        '"""One entry per group of opcodes sharing a slot, an operand shape and a writeback offset."""',
        "",
    ]
    return "\n".join(lines)


def _python_literal(value: Any) -> str:
    """Renders a JSON-derived value as a Python literal, quoting strings the way ruff format does."""
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    return repr(value)


def _check_assembler_constants(data: dict[str, Any]) -> list[str]:
    """Returns a message per scheduling constant in the assembler that disagrees with the source.

    The assembler ships to the board and cannot read the source at runtime, so its capacity constants are written
    out by hand and held against the source here rather than generated from it.
    """
    source = _ASSEMBLER.read_text()
    relative = _ASSEMBLER.relative_to(_REPO)
    problems = []
    for group, spec in data["resources"].items():
        for name, field in spec["asm_constants"].items():
            expected = spec[field]
            # Tolerates the type annotation on each constant, and refuses a name prefixing a longer one.
            match = re.search(rf"^{re.escape(name)}(?![A-Za-z0-9_])\s*(?::[^=\n]+)?=\s*(\d+)", source, re.MULTILINE)
            if match is None:
                problems.append(f"{relative} does not define {name} "
                                f"({_SOURCE.relative_to(_REPO)} {group} resource, {field} = {expected})")
            elif int(match.group(1)) != expected:
                problems.append(f"{relative} {name} = {match.group(1)}, but "
                                f"{_SOURCE.relative_to(_REPO)} {group} resource, {field} = {expected}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
