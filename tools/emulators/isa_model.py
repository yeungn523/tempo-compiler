"""Provides the per-opcode facts the emulator decodes images with.

Run `python3 tools/toolchain/gen_isa_doc.py --write` to generate.
"""

from __future__ import annotations

PROFILE: dict[str, int] = {
    "lane_count": 16,
    "sublane_count": 4,
    "vreg_element_count": 64,
    "vreg_entry_count": 32,
}
"""Array geometry the emulator decodes vector registers and VMEM rows against."""

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
