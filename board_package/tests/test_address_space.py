"""Contains tests for the address-space translation assets and the packaged platform.json reader.

On ZCU104 the two masters reach the same DDR slave at different offsets, so a wrong alias is silent: COPY succeeds
and LAUNCH reads DRAM 2 GiB away.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

# Extends the import path before the runtime imports below resolve, as the package ships the runtime beside the
# tests rather than installing it.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from runtime.address_space import (  # noqa: E402
    ZCU102_ADDRESS_SPACE,
    ZCU104_ADDRESS_SPACE,
    address_space_from_platform_json,
    load_packaged_address_space,
)


class AddressSpaceTest(unittest.TestCase):
    """Verifies the pointer translation performed by the board address-space maps."""

    def test_zcu104_kernel_pointer_aliases_two_gib_in(self) -> None:
        """Verifies that the ZCU104 map translates both ways across the 2 GiB alias split."""
        space = ZCU104_ADDRESS_SPACE
        self.assertEqual(space.to_copy(0x8000_0000), 0x4_8000_0000)
        self.assertEqual(space.to_kernel(0x4_8000_0000), 0x8000_0000)

    def test_zcu102_has_no_alias_split(self) -> None:
        """Verifies that the ZCU102 map leaves the base pointer unshifted."""
        space = ZCU102_ADDRESS_SPACE
        self.assertEqual(space.to_copy(space.kernel_base), space.copy_base)

    def test_out_of_range_pointer_is_rejected(self) -> None:
        """Verifies that a pointer outside the mapped aperture raises."""
        with self.assertRaises(ValueError):
            ZCU104_ADDRESS_SPACE.to_copy(0x1000)


class PlatformJsonTest(unittest.TestCase):
    """Verifies the reader for the platform.json the Vivado build emits beside a bitstream."""

    def test_round_trips_the_zcu104_map(self) -> None:
        """Verifies that a complete platform.json reconstructs the ZCU104 map exactly."""
        want = ZCU104_ADDRESS_SPACE
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_platform_json(
                directory=pathlib.Path(directory),
                facts={
                    "board": "zcu104",
                    "copy_base": want.copy_base,
                    "copy_size": want.copy_size,
                    "kernel_base": want.kernel_base,
                    "kernel_size": want.kernel_size,
                    "kernel_alias_copy_base": want.kernel_alias_base,
                },
            )
            self.assertEqual(address_space_from_platform_json(path), want)

    def test_missing_field_is_an_error_not_a_default(self) -> None:
        """Verifies that an incomplete platform.json raises instead of substituting defaults."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_platform_json(
                directory=pathlib.Path(directory),
                facts={"copy_base": 0, "copy_size": 1},
            )
            with self.assertRaises(ValueError):
                address_space_from_platform_json(path)

    def test_absent_platform_json_falls_back_rather_than_raising(self) -> None:
        """Verifies that a directory carrying no platform.json yields None."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_packaged_address_space(directory))
        self.assertIsNone(load_packaged_address_space("/nonexistent"))

    @staticmethod
    def _write_platform_json(directory: pathlib.Path, facts: dict[str, int | str]) -> pathlib.Path:
        """Writes the given facts to a platform.json inside the directory and returns its path."""
        path = directory / "platform.json"
        path.write_text(json.dumps(facts))
        return path


if __name__ == "__main__":
    unittest.main()
