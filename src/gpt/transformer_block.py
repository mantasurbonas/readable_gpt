"""One Transformer block: attention, then feed-forward, each wrapped in a residual connection."""

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention
from utils.math_utils import normalize_layer


class TransformerBlock:
    """One Transformer block: find relevant information, then process it, preserving the original.

    Built once at load time. The attention and feed-forward objects only hold
    weight matrices that were frozen during training. Nothing here changes
    between predictions — only the token representations fed into calculate()
    are new on each call.
    """

    def __init__(self, block_parameters, attention_head_count):
        # These two objects hold only frozen weight matrices from the trained model.
        # They are built once here, not on every call to calculate().
        self._attention = MultiHeadAttention(block_parameters, attention_head_count)
        self._feed_forward = FeedForward(block_parameters)

        # Layer-norm parameters are also frozen weight tensors (a learned scale
        # and offset vector for each of the two normalizations in this block).
        self._attention_norm = block_parameters.attention_norm
        self._feed_forward_norm = block_parameters.feed_forward_norm

    def calculate(self, token_representations):
        """Attention and the feed-forward network never replace what came in. Each
        one's output is added back to its own input, so information can always
        flow straight through the block:

        input ──────────────────────┐
          ↓                         │
        layer normalization         │
          ↓                         │
        attention                   │
          ↓                         │
        output + original input <───┘
          ↓
          ↓   ──────────────────────┐
          ↓                         │
        layer normalization         │
          ↓                         │
        feed forward                │
          ↓                         │
        output + original input <───┘
          ↓
        result
        """

        # Normalize, attend, then add the result back to the input (residual connection).
        normalized_for_attention = normalize_layer(token_representations, self._attention_norm)
        attention_result = self._attention.calculate(normalized_for_attention)
        after_attention = token_representations + attention_result

        # Normalize, process through the feed-forward network, then add back (residual).
        normalized_for_feed_forward = normalize_layer(after_attention, self._feed_forward_norm)
        feed_forward_result = self._feed_forward.calculate(normalized_for_feed_forward)
        after_feed_forward = after_attention + feed_forward_result

        return after_feed_forward
