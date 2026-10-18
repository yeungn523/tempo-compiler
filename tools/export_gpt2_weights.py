#!/usr/bin/env python3
"""Downloads GPT-2 small's checkpoint and tokenizer and writes them where the board package reads them.

The checkpoint is read straight from its safetensors file, so the export needs neither torch nor transformers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import urllib.request

import numpy as np
from numpy.typing import NDArray

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""The repository's root directory."""

sys.path.insert(0, str(_HERE / "board_package"))

from bf16 import as_bf16_words  # noqa: E402

_REPOSITORY: str = "https://huggingface.co/openai-community/gpt2/resolve/main"
"""Address the checkpoint and the tokenizer files are downloaded from."""

_CHECKPOINT: str = "model.safetensors"
"""Name of the checkpoint file in the repository."""

_TOKENIZER: dict[str, str] = {"vocab.json": "gpt2_vocab.json", "merges.txt": "gpt2_merges.txt"}
"""Name each tokenizer file is written under in the board package, keyed by its name in the repository."""

_WEIGHTS: str = "gpt2_weights.npz"
"""Name the exported weights are written under in the board package."""

_LAYERS: int = 12
"""Transformer layers GPT-2 small stacks."""

_LAYER_NAMES: dict[str, str] = {
    "ln_1.weight": "ln_1.g", "ln_1.bias": "ln_1.b",
    "attn.c_attn.weight": "attn.qkv.w", "attn.c_attn.bias": "attn.qkv.b",
    "attn.c_proj.weight": "attn.proj.w", "attn.c_proj.bias": "attn.proj.b",
    "ln_2.weight": "ln_2.g", "ln_2.bias": "ln_2.b",
    "mlp.c_fc.weight": "mlp.fc.w", "mlp.c_fc.bias": "mlp.fc.b",
    "mlp.c_proj.weight": "mlp.proj.w", "mlp.c_proj.bias": "mlp.proj.b",
}
"""Name the board package gives each per-layer tensor, keyed by its checkpoint name after the layer prefix."""

_SHARED_NAMES: dict[str, str] = {"wte.weight": "wte", "wpe.weight": "wpe", "ln_f.weight": "ln_f.g",
                                 "ln_f.bias": "ln_f.b"}
"""Name the board package gives each tensor every layer shares, keyed by its checkpoint name."""

_DTYPES: dict[str, str] = {"F32": "<f4", "F16": "<f2"}
"""NumPy element type of each safetensors element type the checkpoint may store its tensors in."""


def _export_weights(*, checkpoint: pathlib.Path) -> dict[str, NDArray[np.uint16]]:
    """Reads the checkpoint and returns every weight under the board package's name, as BF16 words.

    Args:
        checkpoint: The downloaded safetensors file.

    Returns:
        Each weight's BF16 container words, keyed by the name the board package reads it under, every per-layer
        name carrying an `h.<layer>.` prefix.

    Raises:
        ValueError: If the checkpoint lacks a weight GPT-2 small carries, or stores one in an unknown element type.
    """
    tensors = _read_safetensors(checkpoint)
    named: dict[str, str] = {source: target for source, target in _SHARED_NAMES.items()}
    for layer in range(_LAYERS):
        named.update({f"h.{layer}.{source}": f"h.{layer}.{target}" for source, target in _LAYER_NAMES.items()})
    missing = sorted(set(named) - set(tensors))
    if missing:
        message = (
            f"Unable to export GPT-2 small from a checkpoint lacking {len(missing)} of its weights, such as "
            f"{missing[0]}. Download the 124M checkpoint, whose layers carry every weight the board package reads."
        )
        raise ValueError(message)
    return {target: as_bf16_words(values=tensors[source]) for source, target in named.items()}


def _read_safetensors(path: pathlib.Path) -> dict[str, NDArray[np.float32]]:
    """Returns every tensor a safetensors file holds, as float32, keyed by its name.

    Args:
        path: The file to read.

    Returns:
        The tensors.

    Raises:
        ValueError: If a tensor is stored in an element type this reader does not convert.
    """
    raw = path.read_bytes()
    (header_bytes,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + header_bytes])
    data = memoryview(raw)[8 + header_bytes:]
    tensors: dict[str, NDArray[np.float32]] = {}
    for name, described in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES.get(described["dtype"])
        if dtype is None:
            message = (
                f"Unable to read tensor {name}, stored as {described['dtype']}. The reader converts "
                f"{', '.join(sorted(_DTYPES))} tensors."
            )
            raise ValueError(message)
        start, end = described["data_offsets"]
        tensors[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(described["shape"]).astype(np.float32)
    return tensors


def _download(*, name: str, cache: pathlib.Path) -> pathlib.Path:
    """Returns one repository file, downloading it into the cache unless an earlier export already did.

    Args:
        name: The file's name in the repository.
        cache: Directory the downloads are kept in.

    Returns:
        The path to the downloaded file.
    """
    destination = cache / name
    if destination.exists():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {_REPOSITORY}/{name}")
    with urllib.request.urlopen(f"{_REPOSITORY}/{name}") as response, partial.open("wb") as stream:
        while chunk := response.read(1 << 20):
            stream.write(chunk)
    # Renames only a complete download, so an interrupted one is fetched again rather than read short.
    partial.rename(destination)
    return destination


def main() -> int:
    """Downloads the checkpoint and the tokenizer, and writes the weights and the tokenizer files.

    Returns:
        Zero once every file is written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(_HERE / "board_package" / "weights"),
                        help="directory the weights and the tokenizer files are written into")
    parser.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "minitpu" / "gpt2"),
                        help="directory the downloads are kept in, so a second export downloads nothing")
    arguments = parser.parse_args()

    output = pathlib.Path(arguments.output).expanduser().resolve()
    cache = pathlib.Path(arguments.cache).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    weights = _export_weights(checkpoint=_download(name=_CHECKPOINT, cache=cache))
    np.savez(output / _WEIGHTS, **weights)
    print(f"wrote {output / _WEIGHTS}  {len(weights)} tensors, {sum(w.nbytes for w in weights.values()) >> 20} MiB")
    for name, written in _TOKENIZER.items():
        (output / written).write_bytes(_download(name=name, cache=cache).read_bytes())
        print(f"wrote {output / written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
