"""Provides the per-opcode scheduling facts the assembler schedules against.

Run `python3 tools/gen_isa_doc.py --write` to generate.
"""

from __future__ import annotations

PROFILE: dict[str, int] = {
    "lane_count": 16,
    "sublane_count": 4,
    "vreg_element_count": 64,
    "vreg_entry_count": 32,
}
"""Array geometry every offset below is quoted against, and every capacity a consumer derives from it."""

RTL_PARAMS: dict[str, int] = {
    "WB_W_ALU": 5,
    "WB_W_SFU": 7,
    "WB_W_REDUCE": 15,
    "WB_W_VLD": 6,
    "WB_W_MPOP_FIRST": 3,
    "WB_W_MPOP_LAST": 6,
}
"""Writeback reservation offset of each sequencer_pkg localparam tb_isa_conformance checks."""

MATRIX_OCCUPANCY: dict[str, int] = {
    "vmatload": 18,
    "vmatpush": 5,
    "vmatpop": 7,
}
"""Cycles each matrix command holds the controller, measured against the RTL."""

MATRIX_PORT_A: tuple[str, ...] = ("vmatload", "vmatpush")
"""Matrix commands owning VREG read port A while they stream."""

MATRIX_RESULT_LATENCY: int = 82
"""Cycles from a vmatpush to its first result leaving the array."""

OPERATIONS: tuple[dict, ...] = (
    {
        "slot": "E",
        "ops": ["vadd", "vsub", "vmul", "vmax", "vmin"],
        "opcodes": [1, 2, 3, 11, 12],
        "reads": ["ra", "rb"],
        "writes": "rd",
        "w": 5,
        "port_a": True,
    },
    {
        "slot": "E",
        "ops": ["vmov"],
        "opcodes": [4],
        "reads": ["ra"],
        "writes": "rd",
        "w": 5,
        "port_a": True,
    },
    {
        "slot": "E",
        "ops": ["vgelu", "vexp", "vrecip", "vrsqrt"],
        "opcodes": [5, 6, 7, 8],
        "reads": ["ra"],
        "writes": "rd",
        "w": 7,
        "port_a": True,
    },
    {
        "slot": "E",
        "ops": ["vredsum", "vredmax"],
        "opcodes": [9, 10],
        "reads": ["ra"],
        "writes": "rd",
        "w": 15,
        "port_a": True,
    },
    {
        "slot": "E",
        "ops": ["vtxin"],
        "opcodes": [13],
        "reads": ["ra"],
        "writes": None,
        "w": None,
        "port_a": True,
    },
    {
        "slot": "E",
        "ops": ["vtxout"],
        "opcodes": [14],
        "reads": [],
        "writes": "rd",
        "w": 4,
        "port_a": False,
    },
    {
        "slot": "X",
        "ops": ["vld"],
        "opcodes": [],
        "reads": [],
        "writes": "vreg",
        "w": 6,
        "port_a": False,
    },
    {
        "slot": "X",
        "ops": ["vst"],
        "opcodes": [],
        "reads": ["vreg"],
        "writes": None,
        "w": None,
        "port_a": True,
    },
    {
        "slot": "M",
        "ops": ["vmatload"],
        "opcodes": [1],
        "reads": ["reg_idx+0..3"],
        "writes": None,
        "w": None,
        "port_a": True,
    },
    {
        "slot": "M",
        "ops": ["vmatpush"],
        "opcodes": [2],
        "reads": ["reg_idx"],
        "writes": None,
        "w": None,
        "port_a": True,
    },
    {
        "slot": "M",
        "ops": ["vmatpop"],
        "opcodes": [3],
        "reads": [],
        "writes": "reg_idx",
        "w": "3..6",
        "port_a": False,
    },
    {
        "slot": "S",
        "ops": ["smovi", "smov_arg", "saddi", "smac", "sshl"],
        "opcodes": [0, 1, 2, 3, 4],
        "reads": ["rs"],
        "writes": "rd",
        "w": None,
        "port_a": False,
    },
    {
        "slot": "F",
        "ops": ["lbegin", "loop_end", "flush_slot", "halt", "vmemswap"],
        "opcodes": [1, 2, 3, 4, 5],
        "reads": [],
        "writes": None,
        "w": None,
        "port_a": False,
    },
)
"""One entry per group of opcodes sharing a slot, an operand shape and a writeback offset."""
