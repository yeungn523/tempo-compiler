"""Provides the images the hand-written builders assembled, frozen when the builders were retired.

The compiled images are held to these, each being the program that ran on the board before a compiler emitted it.
"""

from __future__ import annotations

import pathlib

import numpy as np

_GOLDEN: pathlib.Path = pathlib.Path(__file__).resolve().parent / "golden" / "hand_images.npz"
"""File the frozen images are read from."""


def looped_gemm(column_tiles: int, groups: int, *, activate: bool, loop_blocks: int) -> tuple[int, ...]:
    """Returns the bundle words the hand-written looped GEMM assembled for one shape."""
    return _words(_image(f"looped_gemm_{column_tiles}_{groups}_{loop_blocks}_{int(activate)}"))


def elementwise_add(chunks: int, *, broadcast_right: bool) -> bytes:
    """Returns the image the hand-written elementwise add assembled for one width."""
    return _image(f"elementwise_add_{chunks}_{int(broadcast_right)}")


def layernorm_rows() -> bytes:
    """Returns the image the hand-written LayerNorm assembled over one packed token block."""
    return _image("layernorm_rows_32")


def _image(name: str) -> bytes:
    """Returns one frozen image's bytes.

    Raises:
        KeyError: If no image was frozen under the name.
    """
    frozen = np.load(_GOLDEN)
    if name not in frozen.files:
        message = (
            f"Unable to find the hand-written image {name}. The builders are retired, so only the shapes frozen in "
            f"{_GOLDEN.name} can be compared against."
        )
        raise KeyError(message)
    return frozen[name].tobytes()


def _words(image: bytes) -> tuple[int, ...]:
    """Returns an image's bundle words, each 16 little-endian bytes."""
    return tuple(int.from_bytes(image[at:at + 16], "little") for at in range(0, len(image), 16))
