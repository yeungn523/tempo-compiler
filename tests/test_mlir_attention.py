"""Holds the attention images the MLIR pipeline builds to a float64 reference and to the images codegen/ compiles,
through the emulator."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
"""Stores the path to the repository's root directory."""

sys.path.insert(0, str(_HERE / "board_package"))
sys.path.insert(0, str(_HERE / "tools" / "board"))
sys.path.insert(0, str(_HERE / "tools" / "emulators"))

from bf16 import (_BLOCK_TOKENS, _TILE, _ZERO_TILE_BYTES, as_bf16_words, decode_mask_tiles,  # noqa: E402
                  from_bf16_words, pack_attention_constants)
from build_gpt2_images import _compile_attention, _compile_decode  # noqa: E402
from build_mlir_images import _MASKED, _TOOLS, attention_constants, attention_kernel, compile_kernel  # noqa: E402
from minitpu_emulator import Machine, to_bf16  # noqa: E402

pytestmark = pytest.mark.skipif(not (_TOOLS / "minitpu-opt").exists(), reason="mlir/ is not built")

_HEADS: int = 12
"""Heads the fused tensor carries."""

_HEAD: int = 64
"""Width of one head."""

_MODEL: int = _HEADS * _HEAD
"""Width of the queries, the keys and the values each."""


def _pack(matrix: np.ndarray) -> np.ndarray:
    """Lays a [tokens, features] tensor out as the device holds it: per token block, a zeroed tile and then its
    feature tiles, one row per token."""
    blocks = []
    for block in range(matrix.shape[0] // _BLOCK_TOKENS):
        rows = matrix[block * _BLOCK_TOKENS:(block + 1) * _BLOCK_TOKENS]
        tiles = rows.reshape(_BLOCK_TOKENS, -1, _TILE).transpose(1, 0, 2).reshape(-1)
        blocks += [np.zeros(_TILE * _TILE, dtype=np.float32), tiles]
    return as_bf16_words(values=np.concatenate(blocks))


def _unpack(words: np.ndarray, *, tokens: int, features: int) -> np.ndarray:
    """Reads a tensor packed as `_pack` lays it out back into [tokens, features]."""
    block_words = _TILE * _TILE + features * _BLOCK_TOKENS
    rows = []
    for block in range(tokens // _BLOCK_TOKENS):
        tiles = from_bf16_words(words=words[block * block_words + _TILE * _TILE:(block + 1) * block_words])
        rows.append(tiles.reshape(features // _TILE, _BLOCK_TOKENS, _TILE).transpose(1, 0, 2).reshape(_BLOCK_TOKENS, -1))
    return np.concatenate(rows)[:tokens]


def _reference(query: np.ndarray, key: np.ndarray, value: np.ndarray, *, causal: bool, length: int) -> np.ndarray:
    """Returns every head's attention in float64, each query reading the keys up to its own or the first `length`."""
    output = np.empty_like(query, dtype=np.float64)
    for head in range(_HEADS):
        columns = slice(head * _HEAD, (head + 1) * _HEAD)
        scores = query[:, columns].astype(np.float64) @ key[:length, columns].astype(np.float64).T
        if causal:
            scores = np.where(np.tril(np.ones_like(scores, dtype=bool)), scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        output[:, columns] = (weights / weights.sum(axis=1, keepdims=True)) @ value[:length, columns]
    return output


def _launch(image: bytes, *, constants: np.ndarray, fused: np.ndarray, cache: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Runs one attention image and returns the fused tensor's words after it and the cycles it took."""
    constants_address = 4096
    fused_address = constants_address + constants.nbytes + 4096 + _ZERO_TILE_BYTES
    cache_address = fused_address + fused.nbytes + 4096 + _ZERO_TILE_BYTES
    end = cache_address + (0 if cache is None else cache.nbytes) + 4096
    machine = Machine(np.zeros(end, dtype=np.uint8),
                      arguments=(constants_address, cache_address + _ZERO_TILE_BYTES,
                                 fused_address + _ZERO_TILE_BYTES))
    machine.write_words(constants_address, constants)
    machine.write_words(fused_address, fused)
    if cache is not None:
        machine.write_words(cache_address, cache)
    machine.run(read_image_bytes(image))
    return machine.read_words(fused_address, fused.size), machine.cycles


def read_image_bytes(image: bytes) -> list[int]:
    """Returns an image's bundle words, as read_image returns a file's."""
    return [int.from_bytes(image[offset:offset + 16], "little") for offset in range(0, len(image), 16)]


def _mlir_image(*, keys: int, causal: bool, directory: pathlib.Path) -> bytes:
    """Compiles one attention tile op and returns its image."""
    image = directory / "attention.bin"
    compile_kernel(attention_kernel(keys=keys, causal=causal), image=image)
    return image.read_bytes()


@pytest.mark.parametrize("tokens", [32, 64])
def test_mlir_causal_attention_matches_the_reference(tokens: int, tmp_path: pathlib.Path) -> None:
    """Verifies that each causal attention attends every head within BF16's error of a float64 reference, with one
    token block and two, the most whose unrolled query blocks fit the instruction memory."""
    rng = np.random.default_rng(tokens)
    fused = to_bf16(rng.standard_normal((tokens, 3 * _MODEL)).astype(np.float32))
    constants = np.frombuffer(attention_constants(), dtype="<u2")
    words, _ = _launch(_mlir_image(keys=tokens, causal=True, directory=tmp_path), constants=constants,
                       fused=_pack(fused), cache=None)
    actual = _unpack(words, tokens=tokens, features=3 * _MODEL)[:, :_MODEL]
    expected = _reference(fused[:, :_MODEL], fused[:, _MODEL:2 * _MODEL], fused[:, 2 * _MODEL:], causal=True,
                          length=tokens)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.02


@pytest.mark.parametrize(("keys", "length"), [(64, 1), (64, 49), (256, 200), (1024, 1024)])
def test_mlir_decode_attention_matches_the_reference(keys: int, length: int, tmp_path: pathlib.Path) -> None:
    """Verifies that each decode attention attends the new token over the cached keys up to the cache's length.

    Notes:
        The bound widens with the keys, the output accumulating every key block in BF16 registers as codegen/'s image
        does, which reaches the same 2.5% over 1024 keys.
    """
    rng = np.random.default_rng(keys + length)
    fused = to_bf16(rng.standard_normal((_BLOCK_TOKENS, 3 * _MODEL)).astype(np.float32))
    cache = to_bf16(rng.standard_normal((keys, 2 * _MODEL)).astype(np.float32))
    masks = decode_mask_tiles(keys=keys, length=length).swapaxes(-1, -2)
    constants = pack_attention_constants(initial_maximum=_MASKED, constants=(), masks=masks).reshape(-1)
    words, _ = _launch(_mlir_image(keys=keys, causal=False, directory=tmp_path), constants=constants,
                       fused=_pack(fused), cache=_pack(cache))
    actual = _unpack(words, tokens=_BLOCK_TOKENS, features=3 * _MODEL)[0, :_MODEL]
    expected = _reference(fused[:1, :_MODEL], cache[:, :_MODEL], cache[:, _MODEL:], causal=False, length=length)[0]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < (0.03 if keys > 256 else 0.02)


def test_mlir_attention_beats_codegen(tmp_path: pathlib.Path) -> None:
    """Verifies that GPT-2's causal and decode attention images agree with codegen/'s within BF16's error and take
    fewer cycles."""
    rng = np.random.default_rng(7)
    fused = to_bf16(rng.standard_normal((64, 3 * _MODEL)).astype(np.float32))
    baseline, baseline_constants, _ = _compile_attention(tokens=64, heads=_HEADS)
    mlir_words, mlir_cycles = _launch(_mlir_image(keys=64, causal=True, directory=tmp_path),
                                      constants=np.frombuffer(attention_constants(), dtype="<u2"),
                                      fused=_pack(fused), cache=None)
    baseline_words, baseline_cycles = _launch(baseline, constants=np.frombuffer(baseline_constants, dtype="<u2"),
                                              fused=_pack(fused), cache=None)
    mlir_output = _unpack(mlir_words, tokens=64, features=3 * _MODEL)[:, :_MODEL]
    baseline_output = _unpack(baseline_words, tokens=64, features=3 * _MODEL)[:, :_MODEL]
    assert np.linalg.norm(mlir_output - baseline_output) / np.linalg.norm(baseline_output) < 0.02
    assert mlir_cycles < baseline_cycles

    cache = to_bf16(rng.standard_normal((256, 2 * _MODEL)).astype(np.float32))
    baseline, entry = _compile_decode(keys=256, heads=_HEADS)
    masks = decode_mask_tiles(keys=256, length=200)
    baseline_constants = pack_attention_constants(initial_maximum=entry["entering"], constants=tuple(entry["stated"]),
                                                  masks=masks).reshape(-1)
    mlir_constants = pack_attention_constants(initial_maximum=_MASKED, constants=(),
                                              masks=masks.swapaxes(-1, -2)).reshape(-1)
    _, mlir_cycles = _launch(_mlir_image(keys=256, causal=False, directory=tmp_path), constants=mlir_constants,
                             fused=_pack(fused[:_BLOCK_TOKENS]), cache=_pack(cache))
    _, baseline_cycles = _launch(baseline, constants=baseline_constants, fused=_pack(fused[:_BLOCK_TOKENS]),
                                 cache=_pack(cache))
    assert mlir_cycles < baseline_cycles
