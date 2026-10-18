#!/usr/bin/env python3
"""Print the build identity of the bitstream currently in the PL.

A board keeps whatever was last programmed into it, across reboots of the host
process and across sessions. Every measurement in the README is against one
image, so the first question about any board number is which image produced it --
and nothing else in the package answers it.

    python3 fpga/scripts/read_build_id.py

Prints the 32-bit `build_id` CSR, which a board build stamps from the Git commit
it was built at. Compare it with the identity the README names.
"""

from __future__ import annotations

import pathlib
import sys

PACKAGE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE))


def main() -> int:
    from runtime.abi import AXI_REG
    from runtime.device import minitpuSetDevice
    from runtime.memcpy import _backend

    minitpuSetDevice("minitpu")
    print(f"build_id = 0x{_backend().read32(AXI_REG['build_id']):08X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
