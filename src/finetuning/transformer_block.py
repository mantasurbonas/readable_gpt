"""One Transformer block, able to pass blame back."""

from .feed_forward import FeedForward
from .layer_normalization import LayerNormalization
from .multi_head_attention import MultiHeadAttention


class TransformerBlock:
    """One Transformer block: find relevant information, then process it, preserving the original.

    Same structure as gpt/transformer_block.py. See that file for the diagram
    of the two "output + original input" shortcuts (residual connections).
    """

    def __init__(self, block_parameters, attention_head_count, block_index, tracer):
        self._block_index = block_index
        self._tracer = tracer

        self._attention = MultiHeadAttention(
            block_parameters,
            attention_head_count,
            block_index,
            tracer,
        )
        self._feed_forward = FeedForward(block_parameters, block_index, tracer)

        checkpoint_prefix = f"model/h{block_index}"
        self._attention_norm = LayerNormalization(block_parameters.attention_norm, checkpoint_prefix + "/ln_1")
        self._feed_forward_norm = LayerNormalization(block_parameters.feed_forward_norm, checkpoint_prefix + "/ln_2")

    def calculate(self, token_representations):
        normalized_for_attention = self._attention_norm.calculate(token_representations)
        attention_result = self._attention.calculate(normalized_for_attention)
        after_attention = token_representations + attention_result

        normalized_for_feed_forward = self._feed_forward_norm.calculate(after_attention)
        feed_forward_result = self._feed_forward.calculate(normalized_for_feed_forward)
        after_feed_forward = after_attention + feed_forward_result

        return after_feed_forward

    def pass_blame_back(self, blame_for_output):
        """Walk `calculate()` in reverse.

        Every "output + original input" shortcut sends the blame down BOTH roads:
        straight to the original input, and through the part it wrapped.
        The two shares are added together at the end.
        """
        # Reverse of: after_feed_forward = after_attention + feed_forward_result
        blame_through_feed_forward = self._feed_forward_norm.pass_blame_back(
            self._feed_forward.pass_blame_back(blame_for_output)
        )
        blame_for_after_attention = blame_for_output + blame_through_feed_forward

        # Reverse of: after_attention = token_representations + attention_result
        blame_through_attention = self._attention_norm.pass_blame_back(
            self._attention.pass_blame_back(blame_for_after_attention)
        )
        blame_for_input = blame_for_after_attention + blame_through_attention

        return blame_for_input

    def adjustable_weights(self):
        return (
            self._attention_norm.adjustable_weights()
            + self._attention.adjustable_weights()
            + self._feed_forward_norm.adjustable_weights()
            + self._feed_forward.adjustable_weights()
        )
