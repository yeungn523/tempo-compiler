"""Address-space translation and packaged platform.json handling.

The alias split is the dangerous part of this map: on ZCU104 the two masters
reach the same DDR slave through segments at different offsets, and getting it
wrong is silent -- COPY succeeds, LAUNCH reads DRAM 2 GiB away, and the tile
executes uninitialised memory.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from runtime.address_space import (  # noqa: E402
    ZCU102_ADDRESS_SPACE,
    ZCU104_ADDRESS_SPACE,
    address_space_from_platform_json,
    load_packaged_address_space,
)


class AddressSpaceTest(unittest.TestCase):
    def test_zcu104_kernel_pointer_aliases_two_gib_in(self) -> None:
        space = ZCU104_ADDRESS_SPACE
        self.assertEqual(space.to_copy(0x8000_0000), 0x4_8000_0000)
        self.assertEqual(space.to_kernel(0x4_8000_0000), 0x8000_0000)

    def test_zcu102_has_no_alias_split(self) -> None:
        space = ZCU102_ADDRESS_SPACE
        self.assertEqual(space.to_copy(space.kernel_base), space.copy_base)

    def test_out_of_range_pointer_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ZCU104_ADDRESS_SPACE.to_copy(0x1000)


class PlatformJsonTest(unittest.TestCase):
    """platform.json is emitted beside a bitstream by the Vivado build."""

    def _write(self, directory: pathlib.Path, facts: dict) -> pathlib.Path:
        path = directory / "platform.json"
        path.write_text(json.dumps(facts))
        return path

    def test_round_trips_the_zcu104_map(self) -> None:
        want = ZCU104_ADDRESS_SPACE
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), {
                "board": "zcu104",
                "copy_base": want.copy_base,
                "copy_size": want.copy_size,
                "kernel_base": want.kernel_base,
                "kernel_size": want.kernel_size,
                "kernel_alias_copy_base": want.kernel_alias_base,
            })
            self.assertEqual(address_space_from_platform_json(path), want)

    def test_missing_field_is_an_error_not_a_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(pathlib.Path(tmp), {"copy_base": 0, "copy_size": 1})
            with self.assertRaises(ValueError):
                address_space_from_platform_json(path)

    def test_absent_platform_json_falls_back_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_packaged_address_space(tmp))
        self.assertIsNone(load_packaged_address_space("/nonexistent"))


if __name__ == "__main__":
    unittest.main()
