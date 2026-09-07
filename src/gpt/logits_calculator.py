"""Turn a finished token representation back into a word from the vocabulary."""

import numpy as np


class LogitsCalculator:
    """Score every word in the vocabulary, then pick the winner.

    The same table that turned token ids into 768-dimensional meanings at the
    start is reused in reverse here: comparing a token's representation against
    every embedding in the vocabulary says how well each word fits.
    """

    def __init__(self, token_embeddings):
        self._token_embeddings = token_embeddings

    def find_highest_score_token_id(self, transformer_results):
        # Take only the last row — only the final token's representation is used
        # to predict what comes next; all earlier rows are discarded here.
        last_token_representation = transformer_results[-1]  # (768,)

        # Dot the 768-number representation against every row in the vocabulary table.
        # `.T` transposes token_embeddings from (50257, 768) to (768, 50257) so that
        # (768,) @ (768, 50257) produces one score per vocabulary entry.
        logits = last_token_representation @ self._token_embeddings.T  # (50257,)

        # np.argmax returns the *position* (index) of the largest value, not the
        # value itself — that position is the token id with the highest score.
        best_token_id = np.argmax(logits)

        return int(best_token_id)
