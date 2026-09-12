"""The attention part of a Transformer block.

First we look at one head: each token asks which earlier tokens matter,
then collects information from them. Then we run 12 such heads and
combine what they found.
"""

import numpy as np
from utils.math_utils import linear, softmax

class SingleHeadAttention:
    """One head: let every token collect information from the tokens it cares about."""

    def __init__(self, head_size, head_number):
        self._head_size   = head_size
        self._head_number = head_number

    def calculate(self, projected_q_k_v, token_count):
        """For each token, gather useful information from itself and earlier tokens.

        Every token gets three roles, each a short list of numbers:
        Query — "what am I looking for?"
        Key   — "what kind of information do I contain?"
        Value — "what should I give you if you pay attention to me?"

        We compare Queries to Keys to get relevance percentages, then mix
        Values using those percentages. 
        """

        queries, keys, values = self._slice_head_kqv_from_projection(projected_q_k_v)

        # Step 1 — Compare every Query to every Key.
        # In a result, cell [row i, column j] answers:
        #   "How much should token i care about token j?"
        raw_relevance_scores = queries @ keys.T                # (token_count, token_count)

        # Step 2 — Shrink the scores. With 64 numbers per head, the products
        # in Step 1 get large. Softmax on huge numbers puts almost 100% on one
        # token and ~0% on the rest. Dividing by √64 (= 8) keeps the mix spread out.
        scaled_relevance_scores = raw_relevance_scores / np.sqrt(self._head_size)

        # Step 3 — Hide the future ("causal" = you may only look backward).
        # We ADD a huge negative number to forbidden cells. Allowed cells get +0,
        # so their scores are unchanged.
        causal_mask = self._build_causal_mask(token_count, projected_q_k_v.dtype)
        masked_scores = scaled_relevance_scores + causal_mask

        # Step 4 — Turn scores into mixing weights.
        # Each row sums to 1.0 (think "percentages"). Masked future cells become ~0%.
        attention_weights = softmax(masked_scores)             # (token_count, token_count)

        # Step 5 — Mix Values using those percentages.
        # Example: if token "it" gave 70% to "glass" and 30% to "dropped",
        # this row becomes 0.7 * (glass's Value) + 0.3 * (dropped's Value).
        # So each token now carries a blend of what the earlier tokens offered.
        mixed_values = attention_weights @ values              # (token_count, head_size)

        return mixed_values

    def _slice_head_kqv_from_projection(self, projected_q_k_v):
        """Pick this head's Query, Key, and Value out of the wide 2304-number row.

        One token's projected row looks like three blocks glued together:

            [ 768 Query numbers | 768 Key numbers | 768 Value numbers ]

        This head only needs its own 64 from each block.
        """
        all_queries, all_keys, all_values = np.split(
            projected_q_k_v,
            3,
            axis=1,
        )

        head_start = self._head_number * self._head_size
        head_end = head_start + self._head_size

        queries = all_queries[:, head_start:head_end]
        keys    = all_keys   [:, head_start:head_end]
        values  = all_values [:, head_start:head_end]

        return queries, keys, values

    @staticmethod
    def _build_causal_mask(token_count, dtype):
        """Build the "you may not look forward" mask:

            0  -1e10  -1e10  -1e10
            0      0  -1e10  -1e10
            0      0      0  -1e10
            0      0      0      0

        This gets ADDED to the relevance scores. Allowed positions stay at 0,
        while forbidden ones become so negative that softmax turns them into
        roughly 0% attention.
        """

        # np.tri(n) builds an n×n lower-triangular matrix of ones (diagonal included).
        allowed_positions   = np.tri(token_count, dtype=dtype) # (token_count, token_count)
        forbidden_positions = 1 - allowed_positions
        mask_penalty        = -1e10

        return forbidden_positions * mask_penalty


class MultiHeadAttention:
    """Let every head study the same sentence, then combine what they found."""

    def __init__(self, block, head_count):
        self._qkv_projection    = block.qkv_projection
        self._output_projection = block.attention_output_projection

        # Each head only sees 64 numbers, not the full 768.
        # The one big weight matrix is 2304 wide = 3 parts (Q, K, V) × 768.
        # 2304 / 3 / 12 heads = 64 numbers per head.
        head_size = self._qkv_projection.weight.shape[-1] // 3 // head_count

        self._heads = []
        for head_number in range(head_count):
            self._heads.append(SingleHeadAttention(head_size, head_number))

    def calculate(self, token_representations):
        """Run all attention heads and merge their results."""

        token_count = token_representations.shape[0]

        # Step 1 — One multiply builds Query, Key, and Value for every token.
        # Width triples: 768 → 2304. That is Q, K, and V sitting in one row
        # so we do one matrix multiply instead of three. 
        # Will split them apart for each head next.
        projected_q_k_v = linear(
            token_representations,
            weight=self._qkv_projection.weight,
            bias=self._qkv_projection.bias,
        )                                                      # (token_count, 2304)

        # Step 2 — Run every head on the same sentence.
        head_outputs = []

        for head in self._heads:
            head_outputs.append(head.calculate(projected_q_k_v, token_count))

        # Step 3 — Combine outputs from all the heads.
        return self._combine_heads(head_outputs)

    def _combine_heads(self, head_outputs):
        """Glue head outputs side by side, then let the model mix what each head found.
        """

        # np.hstack stacks arrays horizontally (column-wise).
        # 12 heads × 64 numbers = 768. The first Head occupies columns 0–63, the next Head occupies 64–127, and so on.
        # They are still 12 separate findings sitting in separate groups of columns.
        concatenated = np.hstack(head_outputs)                 # (token_count, 768)

        # After hstack, a row is already 768 numbers, but they still live in
        # separate 64-column neighborhoods (head 0's findings cannot yet affect
        # head 7's columns). This last linear layer remixes the whole row:
        # 768 → 768, learned during training, so what one head noticed can
        # spread into the token's full representation.
        return linear(
            concatenated,
            weight=self._output_projection.weight,
            bias=self._output_projection.bias,
        )                                                       # (token_count, 768)
