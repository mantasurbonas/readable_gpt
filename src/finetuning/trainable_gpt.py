"""A GPT-2 that, besides calculating, can be told how wrong it was and who is to blame."""

from utils.tracer import Tracer

from .adjustable_weights import AdjustableWeights
from .layer_normalization import LayerNormalization
from .logits_calculator import LogitsCalculator
from .require_calculated import require_calculated
from .transformer_block import TransformerBlock


class TrainableGPT:
    """Same constructor as `gpt.GPT`, and it works on the very same NumPy arrays.

    Instead of predicting one next token, `score_every_next_token()` scores every
    vocabulary word at every position, so each guess can be compared with the truth.
    `pass_blame_back()` then spreads the blame for the wrong guesses over every
    weight, ready for the WeightAdjuster.
    """

    def __init__(
        self,
        token_embeddings,
        position_embeddings,
        blocks,
        final_layer_norm,
        attention_head_count,
        context_size,
        tracer=None,
    ):
        self._tracer = tracer if tracer is not None else Tracer()
        self._context_size = context_size

        self._token_embeddings = AdjustableWeights(token_embeddings, "model/wte")
        self._position_embeddings = AdjustableWeights(position_embeddings, "model/wpe")
        self._final_layer_norm = LayerNormalization(final_layer_norm, "model/ln_f")

        # The same word table is used twice: to look up meanings at the start
        # and to score words at the end. One object, so blame from both uses
        # lands in the same place.
        self._logits_calculator = LogitsCalculator(self._token_embeddings, self._tracer)

        self._blocks = []
        for block_index, block_parameters in enumerate(blocks):
            self._blocks.append(
                TransformerBlock(
                    block_parameters,
                    attention_head_count,
                    block_index,
                    self._tracer,
                )
            )

        self._remembered_token_ids = None

    def score_every_next_token(self, token_ids):
        """Return an array of shape (token_count, vocabulary_size)."""
        self._assert_token_ids_valid(token_ids)

        token_count = len(token_ids)

        meanings = self._token_embeddings.values[token_ids]
        positions = self._position_embeddings.values[:token_count]
        token_representations = meanings + positions

        for block in self._blocks:
            token_representations = block.calculate(token_representations)

        normalized_representations = self._final_layer_norm.calculate(token_representations)

        self._remembered_token_ids = list(token_ids)
        return self._logits_calculator.calculate(normalized_representations)

    def pass_blame_back(self, blame_for_logits):
        """Spread the blame for the wrong scores over every weight in the model."""
        token_ids = require_calculated(self._remembered_token_ids, "TrainableGPT")

        blame_for_normalized = self._logits_calculator.pass_blame_back(blame_for_logits)
        blame_for_token_representations = self._final_layer_norm.pass_blame_back(blame_for_normalized)

        for block in reversed(self._blocks):
            blame_for_token_representations = block.pass_blame_back(blame_for_token_representations)

        # The first token representation was "meaning + position", so both tables share the blame.
        # A token id can occur many times in the text, and then its row collects every share.
        self._position_embeddings.add_blame_to_first_rows(len(token_ids), blame_for_token_representations)
        self._token_embeddings.add_blame_to_rows(token_ids, blame_for_token_representations)

    def adjustable_weights(self):
        weights = [self._token_embeddings, self._position_embeddings]
        weights += self._final_layer_norm.adjustable_weights()
        for block in self._blocks:
            weights += block.adjustable_weights()
        return weights

    def forget_blame(self):
        for weights in self.adjustable_weights():
            weights.forget_blame()

    def _assert_token_ids_valid(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids must not be empty.")

        if len(token_ids) > self._context_size:
            raise ValueError(f"Context length {len(token_ids)} exceeds model limit context_size={self._context_size}.")
