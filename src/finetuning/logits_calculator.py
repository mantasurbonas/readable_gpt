"""Score every vocabulary word for every position, able to pass blame back."""

from .require_calculated import require_calculated


class LogitsCalculator:
    """Score every word in the vocabulary, at every position.

    The inference version only scores the last position and picks the winner.
    Training wants the scores of all positions: position i is the model's guess
    for the token that follows token i, and each guess can be compared with the truth.

    The word table is the same one that turned token ids into meanings at the
    start of the model, so it receives blame from both ends.
    """

    def __init__(self, token_embeddings, tracer):
        self._token_embeddings = token_embeddings  # AdjustableWeights, shared with GPT
        self._tracer = tracer
        self._remembered_token_representations = None

    def calculate(self, token_representations):
        self._remembered_token_representations = token_representations

        # (token_count, 768) @ (768, 50257) -> one score per vocabulary word, for every position.
        logits = token_representations @ self._token_embeddings.values.T
        return logits

    def pass_blame_back(self, blame_for_logits):
        token_representations = require_calculated(self._remembered_token_representations, "LogitsCalculator")

        self._token_embeddings.add_blame(blame_for_logits.T @ token_representations)

        blame_for_token_representations = blame_for_logits @ self._token_embeddings.values
        return blame_for_token_representations
