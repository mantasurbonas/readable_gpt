"""A small NumPy implementation of the GPT-2 forward pass."""

from .logits_calculator import LogitsCalculator
from .transformer_block import TransformerBlock
from utils.math_utils import normalize_layer


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
    ):
        self._token_embeddings = token_embeddings
        self._position_embeddings = position_embeddings
        self._final_layer_norm = final_layer_norm
        self._context_size = context_size

        # Build each block once. The TransformerBlock holds MultiHeadAttention
        # and FeedForward objects that carry only frozen weight matrices from the
        # trained model. Nothing in these objects changes between predictions.
        self._blocks = []
        for block_parameters in blocks:
            self._blocks.append(TransformerBlock(block_parameters, attention_head_count))

        # LogitsCalculator also holds only a reference to the frozen weight table.
        self._logits_calculator = LogitsCalculator(token_embeddings)

    def predict_next_token(self, token_ids):
        if not token_ids:
            raise ValueError("token_ids must not be empty.")

        if len(token_ids) > self._context_size:
            raise ValueError(f"Context length {len(token_ids)} exceeds model limit context_size={self._context_size}.")

        token_count = len(token_ids)

        # Index the embedding table with a list of row numbers — NumPy returns
        # those rows in the given order, one per token.
        meanings = self._token_embeddings[token_ids]        # (token_count, 768)

        # [:token_count] takes the first token_count rows of the position table,
        # one row for each position 0, 1, 2, …
        positions = self._position_embeddings[:token_count] # (token_count, 768)

        # "token meaning + token position", added element by element.
        token_representations = meanings + positions        # (token_count, 768)

        # Each block has identical structure but different learned weights.
        # As token_representations pass through successive blocks, each block
        # enriches them with more abstract patterns — early blocks notice
        # surface features (grammar, nearby words), later blocks resolve
        # higher-level meaning (which "Apple" means here, pronoun references, etc.).
        for block in self._blocks:
            token_representations = block.calculate(token_representations)

        # Apply the final normalization before converting to vocabulary scores.
        normalized_representations = normalize_layer(token_representations, self._final_layer_norm)

        return self._logits_calculator.find_highest_score_token_id(normalized_representations)
