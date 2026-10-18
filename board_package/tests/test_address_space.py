"""Contains tests for the address-space translation assets.

On ZCU104 the two masters reach the same DDR slave at different offsets, so a wrong alias is silent: COPY succeeds
and LAUNCH reads DRAM 2 GiB away.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the runtime imports below resolve, as the package ships the runtime beside the
# tests rather than installing it.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The root directory of the deployed board package."""

sys.path.insert(0, str(_HERE / "minitpu_runtime"))

from runtime.address_space import ZCU102_ADDRESS_SPACE, ZCU104_ADDRESS_SPACE  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
