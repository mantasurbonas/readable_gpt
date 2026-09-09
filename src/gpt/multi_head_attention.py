"""The attention part of a Transformer block, from all heads down to one."""

import numpy as np
from utils.math_utils import linear, softmax

class SingleHeadAttention:
    """One head: let every token collect information from the tokens it cares about."""

    def __init__(self, head_size, head_number):
        self._head_size   = head_size
        self._head_number = head_number

    def calculate(self, projected_k_q_v, causal_mask):
        """Run the attention formula for one head.
        For each token we ask: "How relevant is every other token to me?"
        Then we mix their Values according to those relevance percentages.
        """

        queries, keys, values = self._slice_head_kqv_from_projection(projected_k_q_v)

        # Step 1 — Compare every Query to every Key.
        # `@` is matrix multiplication; `.T` transposes keys from (token_count, head_size)
        # to (head_size, token_count) so that queries @ keys.T has shape
        # (token_count, token_count): row i, column j answers
        # "How much should token i care about token j?"
        raw_relevance_scores = queries @ keys.T                # (token_count, token_count)

        # Step 2 — Scale scores so large numbers do not break softmax later.
        scaled_relevance_scores = raw_relevance_scores / np.sqrt(self._head_size)

        # Step 3 — Add the causal mask so that forbidden future positions
        # become "-Infinity".
        masked_scores = scaled_relevance_scores + causal_mask

        # Step 4 — Turn scores into percentages.
        # The forbidden future positions become ~0%.
        attention_weights = softmax(masked_scores)             # (token_count, token_count)

        # Step 5 — Mix each token's Value according to those percentages.
        # `@` here: (token_count, token_count) @ (token_count, head_size) → (token_count, head_size)
        mixed_values = attention_weights @ values              # (token_count, head_size)

        return mixed_values

    def _slice_head_kqv_from_projection(self, projected_k_q_v):
        """Extract this head's Query, Key, and Value columns from the combined projection.

        projected_k_q_v is (token_count, 3 × model_dim), laid out as
        [Q_all | K_all | V_all]. Each section occupies model_dim columns;
        this head's window starts at head_number * head_size within each section.
        """
        model_dim = projected_k_q_v.shape[-1] // 3
        start     = self._head_number * self._head_size
        end       = start + self._head_size

        queries = projected_k_q_v[:,               start : end]
        keys    = projected_k_q_v[:,   model_dim + start : model_dim + end]
        values  = projected_k_q_v[:, 2*model_dim + start : 2*model_dim + end]

        return queries, keys, values

class MultiHeadAttention:
    """Let every head study the same sentence, then combine what they found."""

    def __init__(self, block, head_count):
        self._qkv_projection    = block.qkv_projection
        self._output_projection = block.attention_output_projection

        # head_size is derived from the projection weight: output width / 3 parts / head count.
        # For GPT-2 small: 2304 / 3 / 12 = 64.
        head_size = self._qkv_projection.weight.shape[-1] // 3 // head_count

        self._heads = []
        for head_number in range(head_count):
            self._heads.append(SingleHeadAttention(head_size, head_number))

    def calculate(self, token_representations):
        """Run all attention heads and merge their results."""

        token_count = token_representations.shape[0]

        # Step 1 — One linear layer produces Query, Key, and Value together.
        # The weight matrix triples the width: 768 → 2304 = 3 × 768.
        projected_k_q_v = linear(
            token_representations,
            weight=self._qkv_projection.weight,
            bias=self._qkv_projection.bias,
        )                                                      # (token_count, 2304)

        # Step 2 — Build the "no peeking at the future" mask once for all heads.
        causal_mask = self._build_causal_mask(token_count, token_representations.dtype)

        # Step 3 — Run every head on the same sentence.
        head_outputs = []

        for head in self._heads:
            head_outputs.append(head.calculate(projected_k_q_v, causal_mask))

        # Step 4 — Glue head outputs back together and apply the final projection.
        return self._combine_heads(head_outputs)

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

        Why -1e10 and not -np.inf?
        The allowed cells in forbidden_positions are 0. Multiplying 0 * -np.inf
        produces nan, which would corrupt every row. A very large finite penalty
        like -1e10 gives the same near-zero softmax result without nan.
        """

        # np.tri(n) builds an n×n lower-triangular matrix of ones (diagonal included).
        allowed_positions   = np.tri(token_count, dtype=dtype) # (token_count, token_count)
        forbidden_positions = 1 - allowed_positions
        mask_penalty        = -1e10

        return forbidden_positions * mask_penalty

    def _combine_heads(self, head_outputs):
        """Glue head outputs side by side, then let the model mix what each head found.

        np.hstack stacks arrays horizontally (column-wise) — the inverse of the
        per-head column slicing done in SingleHeadAttention.calculate.
        After concatenation the output projection (a learned linear layer) mixes
        information across heads, so the model can blend what the different heads
        noticed into one coherent representation.
        """

        concatenated = np.hstack(head_outputs)                 # (token_count, 768)

        return linear(
            concatenated,
            weight=self._output_projection.weight,
            bias=self._output_projection.bias,
        )                                                       # (token_count, 768)
