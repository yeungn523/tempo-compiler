#!/usr/bin/env python3
"""Generate the ISA operation table and the RTL conformance package.

`docs/isa_latency.json` is the single source of truth for the implemented
scheduling contract.  Three artifacts are derived from it:

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
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "isa_latency.json"
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


def _load() -> dict:
    return json.loads(SOURCE.read_text())


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
                                f"(docs/isa_latency.json resources.{group}.{field} = {expected})")
            elif int(match.group(1)) != expected:
                problems.append(f"board_package/asm.py {name} = {match.group(1)}, but "
                                f"docs/isa_latency.json resources.{group}.{field} = {expected}")
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
            print(f"STALE: {path} does not match docs/isa_latency.json", file=sys.stderr)
        print("run: python3 tools/gen_isa_doc.py --write", file=sys.stderr)
        return 1
    drift = check_asm_constants(data)
    if drift:
        for message in drift:
            print(f"DRIFT: {message}", file=sys.stderr)
        return 1
    print("ISA doc, conformance package and asm.py constants are in sync "
          "with docs/isa_latency.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
