"""Contains tests for GPT-2's byte-level BPE, held to identifiers GPT-2's own tokenizer assigns."""

from __future__ import annotations

import pathlib
import sys
import unittest

# Extends the import path before the imports below resolve, as the board package is not installed.
_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
"""The board package's directory."""

sys.path.insert(0, str(_HERE))

from tokenizer import Tokenizer  # noqa: E402

_EXPORTED: bool = (_HERE / "gpt2_vocab.json").exists() and (_HERE / "gpt2_merges.txt").exists()
"""Whether the tokenizer has been exported, which tools/export_gpt2_weights.py does on a host."""


@unittest.skipUnless(_EXPORTED, "the tokenizer has not been exported")
class TokenizerTest(unittest.TestCase):
    """Covers encoding and decoding against identifiers GPT-2's tokenizer assigns."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = Tokenizer()

    def test_text_encodes_to_gpt2_s_own_identifiers(self) -> None:
        """Asserts two phrases encode to the identifiers GPT-2's reference tokenizer gives them."""
        self.assertEqual(self.tokenizer.encode("The capital of France is"), [464, 3139, 286, 4881, 318])
        self.assertEqual(self.tokenizer.encode("Hello, world!"), [15496, 11, 995, 0])

    def test_every_text_decodes_back_to_itself(self) -> None:
        """Asserts text with whitespace runs, symbols and non-ASCII characters survives a round trip."""
        for text in ("  multiple   spaces\nand\ttabs", "naïve café — 東京", "snake_case_name x__y", "It's 2026."):
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
