#!/usr/bin/env python3
"""Generate the ISA operation table and the RTL conformance package.

`compiler/minitpuisa.td` is the single source of truth for the implemented
scheduling contract.  It is read through `llvm-tblgen --dump-json`, so nothing
here needs an MLIR build; the ODS operation definitions that will eventually
join it are what needs `mlir-tblgen`, and they generate C++, not these.  Three
artifacts are derived from it:

  docs/ISA_AND_INTERFACES.md   the operation table, between GENERATED markers
  tb/isa_latency_pkg.sv        constants tb_isa_conformance.sv checks the
                               sequencer's own localparams against
  board_package/isa_model.py   the per-opcode scheduling facts asm.py reads

The surrounding prose in the markdown is hand-written and is never touched.

    python3 tools/gen_isa_doc.py --write    regenerate all three artifacts
    python3 tools/gen_isa_doc.py --check    exit non-zero if any is stale
"""

from __future__ import annotations

import argparse
import re
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "compiler" / "minitpuisa.td"
DOC = REPO / "docs" / "ISA_AND_INTERFACES.md"
PKG = REPO / "tb" / "isa_latency_pkg.sv"
MODEL = REPO / "board_package" / "isa_model.py"

BEGIN = "<!-- BEGIN GENERATED: tools/gen_isa_doc.py -->"
END = "<!-- END GENERATED -->"

HEADER = "Run `python3 tools/gen_isa_doc.py --write` to generate."
"""Line every generated artifact opens with, carrying the command that writes it and nothing else."""

COLUMNS = (
    "Physical slot / operation",
    "VREG reads",
    "VREG writes",
    "W",
    "Earliest consumer L",
    "Issue interval / structural requirement",
)


TBLGEN_CANDIDATES = ("llvm-tblgen", "llvm-tblgen-18", "llvm-tblgen-17", "llvm-tblgen-19",
                     "llvm-tblgen-20")
"""Binaries tried in order when $LLVM_TBLGEN does not name one."""


def _tblgen() -> str:
    """Returns the llvm-tblgen to read the SSOT with.

    Raises:
        SystemExit: If none is on PATH, naming the environment variable that overrides the search.
    """
    override = os.environ.get("LLVM_TBLGEN")
    if override:
        return override
    for candidate in TBLGEN_CANDIDATES:
        found = shutil.which(candidate)
        if found is not None:
            return found
    raise SystemExit(
        f"Unable to read {SOURCE.relative_to(REPO)} without llvm-tblgen, which is not on PATH under any of "
        f"{', '.join(TBLGEN_CANDIDATES)}. Install an LLVM package or set LLVM_TBLGEN to the binary."
    )


def _records() -> dict:
    """Returns the TableGen record dump of the SSOT.

    Raises:
        SystemExit: If tblgen rejects the file, carrying its own diagnostic rather than a summary of one.
    """
    completed = subprocess.run([_tblgen(), "--dump-json", str(SOURCE)],
                               capture_output=True, text=True)
    if completed.returncode:
        raise SystemExit(f"llvm-tblgen rejected {SOURCE.relative_to(REPO)}:\n{completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _referenced(records: dict, entries: list) -> list:
    """Resolves a field holding a list of record references to the records themselves."""
    return [records[entry["def"]] for entry in entries]


def _ordered(records: dict, cls: str, field: str) -> list:
    """Returns every record of `cls`, in the order `kIsaOrder` names them.

    TableGen's own record listing is sorted by record name rather than by definition order, and the order these
    appear in is part of the contract, so the SSOT names it outright.

    Raises:
        SystemExit: If `kIsaOrder` omits a record. It is refused rather than dropped, because a mnemonic missing
            from the generated model is a hazard the scheduler cannot then see.
    """
    listed = _referenced(records, records["kIsaOrder"][field])
    named = {record["!name"] for record in listed}
    missing = [name for name in records["!instanceof"].get(cls, ()) if name not in named]
    if missing:
        plural = "s" if len(missing) > 1 else ""
        raise SystemExit(
            f"{SOURCE.relative_to(REPO)} defines {cls} record{plural} {', '.join(missing)} that "
            f"kIsaOrder.{field} does not list, so {'they' if plural else 'it'} would not reach the "
            f"generated artifacts. Add {'them' if plural else 'it'} to the list."
        )
    return listed


def _writeback(record: dict):
    """Returns the writeback offset as the artifacts carry it: absent, one offset, or a range of beats.

    An uninitialised offset states that the group reserves no write port at all, which is a different claim from
    reserving it at cycle zero, so it stays distinguishable from a value here.
    """
    if record["writeback_offset"] is None:
        return None
    if record["writeback_last_offset"] is None:
        return record["writeback_offset"]
    return f"{record['writeback_offset']}..{record['writeback_last_offset']}"


def _operations(records: dict) -> list:
    """Returns one entry per operation group, shaped as the renderers below read it.

    Notes:
        An empty field is omitted rather than emitted, because `render_model` selects on whether a group names
        opcodes at all and `_consumer` on whether it overrides its consumer offset.
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
        for key, value in (("opcodes", record["opcodes"]), ("rtl_enum", record["rtl_enumerators"]),
                           ("rtl_param", record["rtl_parameter"]),
                           ("consumer_override", record["consumer_override"])):
            if value:
                entry[key] = value
        groups.append(entry)
    return groups


def _matrix(records: dict) -> dict:
    """Returns the matrix commands' occupancy, result latency and port-A ownership."""
    commands = _ordered(records, "MatrixCommand", "matrix_commands")
    return {
        "occupancy": {command["mnemonic"]: command["occupancy_cycles"] for command in commands},
        "result_latency": {command["mnemonic"]: command["result_latency_cycles"] for command in commands
                           if command["result_latency_cycles"] is not None},
        "port_a_held_by": [command["mnemonic"] for command in commands if command["holds_read_port_a"]],
    }


def _resources(records: dict) -> dict:
    """Returns each capacity's values, RTL parameter, asm.py constants and note."""
    resources = {}
    for record in _ordered(records, "Resource", "resources"):
        spec = {value["name"]: value["value"]
                for value in _referenced(records, record["values"])}
        if record["rtl_parameter"]:
            spec["rtl_param"] = record["rtl_parameter"]
        spec["asm_constants"] = {constant["name"]: constant["resource_field"]
                                 for constant in _referenced(records, record["asm_constants"])}
        if record["note"]:
            spec["note"] = record["note"]
        resources[record["group"]] = spec
    return resources


def _profile(records: dict) -> dict:
    """Returns the array geometry the writeback offsets are quoted against."""
    record = records["kMiniTpuProfile"]
    return {"num_lanes": record["lane_count"], "num_sublanes": record["sublane_count"],
            "vreg_entries": record["vreg_entry_count"], "vreg_elements": record["vreg_element_count"],
            "note": record["note"]}


def _load() -> dict:
    """Returns the SSOT's records reshaped into the mapping every renderer below reads."""
    records = _records()
    return {
        "profile": _profile(records),
        "operations": _operations(records),
        "rtl_params": {record["name"]: record["value"]
                       for record in _ordered(records, "RtlParameter", "rtl_parameters")},
        "matrix": _matrix(records),
        "resources": _resources(records),
    }


def _consumer(entry: dict) -> str:
    """L = W + 1, except where the entry declares its own convention."""
    if "consumer_override" in entry:
        return entry["consumer_override"]
    w = entry["w"]
    if w is None:
        return "—"
    if isinstance(w, str):  # a range such as "3..6" retires at its last beat
        return str(int(w.split("..")[1]) + 1)
    return str(w + 1)


def _w_cell(entry: dict) -> str:
    return "—" if entry["w"] is None else str(entry["w"])


def render_table(data: dict) -> str:
    rows = ["| " + " | ".join(COLUMNS) + " |",
            "| --- | --- | --- | ---: | ---: | --- |"]
    for entry in data["operations"]:
        # Mnemonics render as code; prose descriptions stay plain.
        names = ", ".join(
            f"`{o}`" if o.replace("_", "").isalnum() else o for o in entry["ops"])
        rows.append("| " + " | ".join([
            f"{entry['slot']}: {names}",
            entry["vreg_reads"],
            entry["vreg_writes"],
            _w_cell(entry),
            _consumer(entry),
            entry["issue"],
        ]) + " |")

    # The matrix commands carry two different facts per command and the resources are capacities a
    # schedule can exceed, not latencies. Both were in the SSOT and neither reached the published
    # table, which is how the output FIFO's depth came to be known only to the RTL.
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
    for group, spec in data.get("resources", {}).items():
        if group.startswith("_"):
            continue
        field = next(iter(spec.get("asm_constants", {}).values()), None)
        rows.append("| " + " | ".join([
            group.replace("_", " "),
            str(spec[field]) if field else "—",
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


def render_pkg(data: dict) -> str:
    lines = [
        f"// {HEADER}",
        "",
        "package isa_latency_pkg;",
    ]
    params = dict(data["rtl_params"])
    for group, spec in data.get("resources", {}).items():
        if group.startswith("_") or "rtl_param" not in spec:
            continue
        field = next(iter(spec.get("asm_constants", {}).values()), None)
        if field is not None:
            params[spec["rtl_param"]] = spec[field]
    for name, value in params.items():
        lines.append(f"  localparam int unsigned DOC_{name} = {value};")
    for name, value in opcode_values(data).items():
        lines.append(f"  localparam int unsigned DOC_{name} = {value};")
    lines += ["endpackage : isa_latency_pkg", ""]
    return "\n".join(lines)


def opcode_values(data: dict) -> dict:
    """Returns the numeric value of every opcode the SSOT names, keyed by its sequencer_pkg enum member."""
    values = {}
    for entry in data["operations"]:
        for name, value in zip(entry.get("rtl_enum", ()), entry.get("opcodes", ())):
            values[name] = value
    return values


def _python_literal(value) -> str:
    """Renders a JSON-derived value as a Python literal, quoting strings the way ruff format does."""
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    return repr(value)


def render_model(data: dict) -> str:
    """Returns the generated Python module carrying the per-opcode scheduling facts asm.py reads."""
    groups = []
    for entry in data["operations"]:
        if "opcodes" not in entry and entry["slot"] != "X":
            continue
        groups.append({
            "slot": entry["slot"],
            "ops": entry["ops"],
            "opcodes": entry.get("opcodes", []),
            "reads": entry["reads"],
            "writes": entry["writes"],
            "w": entry["w"],
            "port_a": entry["port_a"],
        })
    lines = [
        '"""Provides the per-opcode scheduling facts the assembler schedules against.',
        "",
        HEADER,
        '"""',
        "",
        "from __future__ import annotations",
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
        "MATRIX_PORT_A: tuple[str, ...] = "
        + repr(tuple(data["matrix"]["port_a_held_by"])).replace("'", '"'),
        '"""Matrix commands owning VREG read port A while they stream."""',
        "",
        f"MATRIX_RESULT_LATENCY: int = {data['matrix']['result_latency']['vmatpush']}",
        '"""Cycles from a vmatpush to its first result leaving the array."""',
        "",
        "OPERATIONS: tuple[dict, ...] = (",
    ]
    for group in groups:
        lines.append("    {")
        for key, value in group.items():
            lines.append(f"        {_python_literal(key)}: {_python_literal(value)},")
        lines.append("    },")
    lines += [
        ")",
        '"""One entry per group of opcodes sharing a slot, an operand shape and a writeback offset."""',
        "",
    ]
    return "\n".join(lines)


def splice(doc_text: str, table: str) -> str:
    if BEGIN not in doc_text or END not in doc_text:
        raise SystemExit(
            f"{DOC} is missing the {BEGIN} / {END} markers around its table"
        )
    head, rest = doc_text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{table}\n{END}{tail}"


def check_asm_constants(data: dict) -> list:
    """Returns a message per scheduling constant in asm.py that disagrees with the SSOT.

    asm.py ships to the board and cannot read this file at runtime, so the constants are written
    out there and checked here instead of generated. That is the point: the MXU output FIFO's depth
    lived only in the RTL, the scheduler carried no notion of it at all, and a kernel that
    overflowed it on every step reached silicon returning plausible wrong numbers.
    """
    source = (REPO / "board_package" / "asm.py").read_text()
    problems = []
    for group, spec in data.get("resources", {}).items():
        if group.startswith("_"):
            continue
        for name, field in spec.get("asm_constants", {}).items():
            expected = spec[field]
            # Tolerates the type annotation on each constant, and refuses a name prefixing a longer one.
            match = re.search(rf"^{re.escape(name)}(?![A-Za-z0-9_])\s*(?::[^=\n]+)?=\s*(\d+)",
                              source, re.MULTILINE)
            if match is None:
                problems.append(f"board_package/asm.py does not define {name} "
                                f"(compiler/minitpuisa.td {group} resource, {field} = {expected})")
            elif int(match.group(1)) != expected:
                problems.append(f"board_package/asm.py {name} = {match.group(1)}, but "
                                f"compiler/minitpuisa.td {group} resource, {field} = {expected}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.write == args.check:
        ap.error("pass exactly one of --write / --check")

    data = _load()
    want_doc = splice(DOC.read_text(), render_table(data))
    want_pkg = render_pkg(data)
    want_model = render_model(data)

    if args.write:
        DOC.write_text(want_doc)
        PKG.write_text(want_pkg)
        MODEL.write_text(want_model)
        print(f"wrote {DOC.relative_to(REPO)}, {PKG.relative_to(REPO)} "
              f"and {MODEL.relative_to(REPO)}")
        return 0

    stale = [p.relative_to(REPO)
             for p, want in ((DOC, want_doc), (PKG, want_pkg), (MODEL, want_model))
             if not p.exists() or p.read_text() != want]
    if stale:
        for path in stale:
            print(f"STALE: {path} does not match compiler/minitpuisa.td", file=sys.stderr)
        print("run: python3 tools/gen_isa_doc.py --write", file=sys.stderr)
        return 1
    drift = check_asm_constants(data)
    if drift:
        for message in drift:
            print(f"DRIFT: {message}", file=sys.stderr)
        return 1
    print("ISA doc, conformance package and asm.py constants are in sync "
          "with compiler/minitpuisa.td")
    return 0


if __name__ == "__main__":
    sys.exit(main())
