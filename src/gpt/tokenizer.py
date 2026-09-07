"""Translate between human-readable text and GPT-2 token identifiers."""

from functools import lru_cache

import regex


class Tokenizer:
    """OpenAI's byte-level byte-pair encoding tokenizer."""

    def __init__(self, vocabulary, merge_rules, errors="replace"):
        self._encoder = vocabulary
        self._decoder = {token_id: token for token, token_id in vocabulary.items()}
        self._errors = errors

        self._byte_encoder = self._bytes_to_unicode()
        self._byte_decoder = {
            character: byte for byte, character in self._byte_encoder.items()
        }
        # merge_rules is ordered by priority: the lowest number merges first.
        self._merge_priorities = {
            pair: priority for priority, pair in enumerate(merge_rules)
        }
        self._cache = {}
        self._pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def encode(self, text):
        """Split text into byte-pair encoded token identifiers."""
        token_ids = []

        for text_piece in self._pattern.findall(text):
            byte_encoded_piece = "".join(
                self._byte_encoder[byte] for byte in text_piece.encode("utf-8")
            )
            token_ids.extend(
                self._encoder[token]
                for token in self._apply_merge_rules(byte_encoded_piece).split(" ")
            )

        return token_ids

    def decode(self, token_ids):
        """Turn token identifiers back into text."""
        byte_encoded_text = "".join(
            self._decoder[token_id] for token_id in token_ids
        )
        return bytearray(
            self._byte_decoder[character] for character in byte_encoded_text
        ).decode("utf-8", errors=self._errors)

    @staticmethod
    @lru_cache
    def _bytes_to_unicode():
        visible_bytes = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        unicode_values = visible_bytes[:]
        next_unicode_value = 2**8

        for byte in range(2**8):
            if byte not in visible_bytes:
                visible_bytes.append(byte)
                unicode_values.append(next_unicode_value)
                next_unicode_value += 1

        return dict(zip(visible_bytes, map(chr, unicode_values)))

    @staticmethod
    def _pairs(word):
        """Return all adjacent symbol pairs in a token."""
        return set(zip(word, word[1:]))

    def _apply_merge_rules(self, token):
        if token in self._cache:
            return self._cache[token]

        word = tuple(token)
        pairs = self._pairs(word)
        if not pairs:
            return token

        while True:
            pair = min(
                pairs,
                key=lambda candidate: self._merge_priorities.get(
                    candidate, float("inf")
                ),
            )
            if pair not in self._merge_priorities:
                break

            first, second = pair
            merged_word = []
            index = 0

            while index < len(word):
                try:
                    first_index = word.index(first, index)
                    merged_word.extend(word[index:first_index])
                    index = first_index
                except ValueError:
                    merged_word.extend(word[index:])
                    break

                if (
                    word[index] == first
                    and index < len(word) - 1
                    and word[index + 1] == second
                ):
                    merged_word.append(first + second)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1

            word = tuple(merged_word)
            if len(word) == 1:
                break
            pairs = self._pairs(word)

        encoded_token = " ".join(word)
        self._cache[token] = encoded_token
        return encoded_token
