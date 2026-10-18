"""Provides GPT-2's byte-level BPE, turning text into the tokens the model reads and its tokens back into text."""

from __future__ import annotations

import json
import pathlib
import re

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""The board package's directory, whose weights directory holds the exported vocabulary and merges."""

_PATTERN: re.Pattern[str] = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
"""GPT-2's pre-tokenizer, splitting text into contractions, letter runs, digit runs, symbol runs and whitespace. The
letter class is every word character but digits and underscores, which is `re`'s spelling of Unicode letters."""


class Tokenizer:
    """Encodes text into GPT-2's tokens and decodes tokens back into text.

    Args:
        vocabulary: The exported `vocab.json`, mapping each token's byte-level string to its identifier, or None for
            the one in this package's weights directory.
        merges: The exported `merges.txt`, listing BPE merges by priority, or None for the one in
            this package's weights directory.

    Attributes:
        _encoder: Identifier of each token's byte-level string.
        _decoder: Byte-level string of each identifier.
        _ranks: Priority of each merge, lower merging first.
        _bytes: The printable character each byte is spelled as.
        _characters: The byte each printable character spells.
        _cache: The merged tokens of every word already encoded.
    """

    def __init__(self, vocabulary: pathlib.Path | None = None, merges: pathlib.Path | None = None) -> None:
        vocabulary = _HERE / "weights" / "gpt2_vocab.json" if vocabulary is None else vocabulary
        merges = _HERE / "weights" / "gpt2_merges.txt" if merges is None else merges
        for path in (vocabulary, merges):
            if not path.exists():
                message = (
                    f"Unable to read GPT-2's tokenizer from {path}. Run tools/board/export_gpt2_weights.py on a host, which "
                    f"writes it there, and deploy the package again."
                )
                raise FileNotFoundError(message)
        self._encoder: dict[str, int] = json.loads(vocabulary.read_text(encoding="utf-8"))
        self._decoder: dict[int, str] = {identifier: text for text, identifier in self._encoder.items()}
        pairs = [tuple(line.split()) for line in merges.read_text(encoding="utf-8").splitlines()[1:] if line]
        self._ranks: dict[tuple[str, str], int] = {pair: rank for rank, pair in enumerate(pairs)}
        self._bytes: dict[int, str] = _byte_characters()
        self._characters: dict[str, int] = {character: byte for byte, character in self._bytes.items()}
        self._cache: dict[str, list[str]] = {}

    def encode(self, text: str) -> list[int]:
        """Returns the tokens one text encodes to."""
        tokens: list[int] = []
        for word in _PATTERN.findall(text):
            spelled = "".join(self._bytes[byte] for byte in word.encode("utf-8"))
            tokens.extend(self._encoder[piece] for piece in self._merged(spelled))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Returns the text tokens decode to, replacing any byte sequence that is not valid UTF-8."""
        spelled = "".join(self._decoder[token] for token in tokens)
        return bytes(self._characters[character] for character in spelled).decode("utf-8", errors="replace")

    def _merged(self, word: str) -> list[str]:
        """Returns one byte-level word split into the vocabulary pieces BPE merges it into."""
        if word in self._cache:
            return self._cache[word]
        pieces = list(word)
        while len(pieces) > 1:
            ranked = [(self._ranks.get(pair, len(self._ranks)), index)
                      for index, pair in enumerate(zip(pieces, pieces[1:]))]
            rank, index = min(ranked)
            if rank == len(self._ranks):
                break
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
        self._cache[word] = pieces
        return pieces


def _byte_characters() -> dict[int, str]:
    """Returns the printable character GPT-2 spells each byte as, so every byte has a vocabulary symbol."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    characters = {byte: chr(byte) for byte in printable}
    shifted = 0
    for byte in range(256):
        if byte not in characters:
            characters[byte] = chr(256 + shifted)
            shifted += 1
    return characters
