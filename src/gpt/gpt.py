"""A small NumPy implementation of the GPT-2 forward pass."""

from .logits_calculator import LogitsCalculator
from .transformer_block import TransformerBlock
from utils.math_utils import normalize_layer
from utils.tracer import Tracer


class GPT:
    """Predict the next token using pretrained GPT-2 parameters."""

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
        self._token_embeddings = token_embeddings
        self._position_embeddings = position_embeddings
        self._final_layer_norm = final_layer_norm
        self._context_size = context_size

        self._logits_calculator = LogitsCalculator(
            token_embeddings,
            self._tracer,
        )

        # Build each block once. 
        # The TransformerBlock holds MultiHeadAttention and FeedForward objects 
        #   that carry only frozen weight matrices from the trained model. 
        # Nothing in these objects changes between predictions.
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

    def predict_next_token(self, token_ids):
        self._assert_token_ids_valid(token_ids)

        token_count = len(token_ids)

        # Index the embedding table with a list of row numbers — NumPy returns
        # those rows in the given order, one per token.
        meanings = self._token_embeddings[token_ids]        # (token_count, 768)
        self._tracer.trace("input token embeddings are", meanings)

        # [:token_count] takes the first token_count rows of the position table,
        # one row for each position 0, 1, 2, …
        positions = self._position_embeddings[:token_count] # (token_count, 768)
        self._tracer.trace("position embeddings are", positions)

        # "token meaning + token position", added element by element.
        token_representations = meanings + positions        # (token_count, 768)
        self._tracer.trace("input token embedding + position are", token_representations)

        # Each block has identical structure but different learned weights.
        # As token_representations pass through successive blocks, each block
        # enriches them with more abstract patterns — early blocks notice
        # surface features (grammar, nearby words), later blocks resolve
        # higher-level meaning (which "Apple" means here, pronoun references, etc.).
        for block in self._blocks:
            token_representations = block.calculate(token_representations)
            self._tracer.trace("next block completed, tokens are", token_representations)

        # Apply the final normalization now.
        normalized_representations = normalize_layer(token_representations, self._final_layer_norm)
        self._tracer.trace("all blocks completed, token representations are", normalized_representations)

        # The next token is predicted based on the very last token_representation only:
        last_token_representation = normalized_representations[-1]
        self._tracer.trace("last token representation for lookup", last_token_representation)

        # Now the logits_calculator can locate the most suitable token from all the known embeddings
        return self._logits_calculator.find_most_suitable_token_id(
            last_token_representation,
            token_ids,
        )

    def _assert_token_ids_valid(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids must not be empty.")

        if len(token_ids) > self._context_size:
            raise ValueError(f"Context length {len(token_ids)} exceeds model limit context_size={self._context_size}.")
