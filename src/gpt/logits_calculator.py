"""Turn a finished token representation back into a word from the vocabulary."""

import numpy as np


class LogitsCalculator:
    """Score every word in the vocabulary, then pick the winner.

    The same table that turned token ids into 768-dimensional meanings at the
    start is reused in reverse here: comparing a token's representation against
    every embedding in the vocabulary says how well each word fits.
    """

    def __init__(self, token_embeddings, tracer):
        self._token_embeddings = token_embeddings
        self._tracer = tracer

    def find_most_suitable_token_id(self, last_token_representation, source_token_ids):
        # Dot the 768-number representation against every row in the vocabulary table.
        # `.T` transposes token_embeddings from (50257, 768) to (768, 50257) so that
        # (768,) @ (768, 50257) produces one score per vocabulary entry.
        logits = last_token_representation @ self._token_embeddings.T  # (50257,)

        # np.argmax returns the *position* (index) of the largest value, not the
        # value itself — that position is the token id with the highest score.
        best_token_id = int(np.argmax(logits))

        return best_token_id