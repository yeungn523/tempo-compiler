#!/usr/bin/env python3
"""Writes the bundle streams codegen/asm.py schedules for every GPT-2 image, which the MLIR scheduler is held to.

Each image yields four streams: what its emitter placed, what the placement pass handed to `schedule`, what `schedule`
returned, and what `pack` makes of the emitter's stream whether or not the image is packed, which holds the MLIR packer
to every region shape the images carry. A recorded stream is matched to its image by content, so an attempt that overflowed IRAM and was
retried under another placement is never mistaken for the image the board runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from dataclasses import dataclass

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "tools" / "board"))

from codegen import asm  # noqa: E402

_STREAMS: tuple[str, ...] = ("emitted", "placed", "scheduled", "packed")
"""Name of each stream an image yields."""


@dataclass(frozen=True, slots=True)
class _Recording:
    """Holds the streams the assembler produced for one kernel.

    Attributes:
        placement: The re-placement pass the kernel was assembled under, empty for the emitter's own placement.
        emitted: The bundle words the emitter placed.
        placed: The bundle words the placement pass handed to `schedule`.
        scheduled: The bundle words `schedule` returned.
        packed: The bundle words `pack` makes of the emitted ones.
    """

    placement: str
    emitted: tuple[int, ...]
    placed: tuple[int, ...]
    scheduled: tuple[int, ...]
    packed: tuple[int, ...]


def main() -> int:
    """Builds every GPT-2 image while recording the assembler's streams, then writes each image's four streams.

    Returns:
        Zero once every image's streams are written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_HERE / "mlir" / "test" / "Inputs" / "schedule"),
                        help="Directory the streams are written into.")
    arguments = parser.parse_args()

    recordings: list[_Recording] = []

    def recording_scheduled(name: str, bundles: tuple[int, ...], placement: str) -> tuple[int, ...]:
        """Assembles one kernel exactly as `asm._scheduled` does, bypassing its memo, and records every stream."""
        placed = list(bundles)
        if placement:
            placed = {"place": asm.place, "pack": asm.pack}[placement](placed)
        scheduled = tuple(asm.schedule(list(placed)))
        recordings.append(_Recording(placement=placement, emitted=tuple(bundles), placed=tuple(placed),
                                     scheduled=scheduled, packed=tuple(asm.pack(list(bundles)))))
        return scheduled

    # Replaces the memoised entry point AsmKernel.assemble resolves at call time, so every image is scheduled afresh.
    asm._scheduled = recording_scheduled
    import build_gpt2_images  # noqa: PLC0415  -- imported after the patch, so it binds nothing stale

    with tempfile.TemporaryDirectory() as images:
        sys.argv = ["build_gpt2_images.py", "--output", images]
        build_gpt2_images.main()
        written = pathlib.Path(arguments.output).expanduser().resolve()
        written.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for image in sorted(pathlib.Path(images).glob("*.bin")):
            if image.stem.endswith("_constants"):
                continue
            matched = _match(image.read_bytes(), recordings)
            for stream in _STREAMS:
                words = getattr(matched, stream)
                (written / f"{image.stem}.{stream}.bin").write_bytes(
                    b"".join(word.to_bytes(asm.BUNDLE_BYTES, "little") for word in words))
            manifest[image.stem] = {"placement": matched.placement, "bundles": len(matched.scheduled)}
        (written / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} images' streams into {written}")
    return 0


def _match(image: bytes, recordings: list[_Recording]) -> _Recording:
    """Returns the recording whose scheduled stream, padded as the loader pads it, is the image.

    Args:
        image: The image's bytes.
        recordings: Every kernel the assembler scheduled while the images were built.

    Returns:
        The recording the image came from.

    Raises:
        ValueError: If no recording reproduces the image.
    """
    for recorded in recordings:
        words = list(recorded.scheduled)
        while len(words) % 4:
            words.append(words[-1])
        if b"".join(word.to_bytes(asm.BUNDLE_BYTES, "little") for word in words) == image:
            return recorded
    message = f"Unable to match an image of {len(image)} bytes. It must be one a recorded schedule produced."
    raise ValueError(message)


if __name__ == "__main__":
    raise SystemExit(main())
