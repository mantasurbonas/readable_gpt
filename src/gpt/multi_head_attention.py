"""The attention part of a Transformer block, from all heads down to one."""

import numpy as np
from utils.math_utils import linear, softmax

class SingleHeadAttention:
    """One head: let every token collect information from the tokens it cares about."""

    def __init__(self, queries, keys, values):
        self._queries = queries
        self._keys = keys
        self._values = values

    def calculate(self, causal_mask):
        """Run the attention formula for one head.
        For each token we ask: "How relevant is every other token to me?"
        Then we mix their Values according to those relevance percentages.
        """

        # .shape[-1] reads the size of the last axis — here it's the column count,
        # which is head_size (64 for GPT-2 small).
        head_size = self._queries.shape[-1]

        # Step 1 — Compare every Query to every Key.
        # `@` is matrix multiplication; `.T` transposes keys from (token_count, 64)
        # to (64, token_count) so that queries @ keys.T has shape
        # (token_count, token_count): row i, column j answers
        # "How much should token i care about token j?"
        raw_relevance_scores = self._queries @ self._keys.T  # (token_count, token_count)

        # Step 2 — Scale scores so large numbers do not break softmax later.
        scaled_relevance_scores = raw_relevance_scores / np.sqrt(head_size)

        # Step 3 — Add the causal mask, then turn scores into percentages.
        # Forbidden future positions become ~0% after softmax.
        masked_scores = scaled_relevance_scores + causal_mask

        attention_weights = softmax(masked_scores)           # (token_count, token_count)

        # Step 4 — Mix each token's Value according to those percentages.
        # `@` here: (token_count, token_count) @ (token_count, 64) → (token_count, 64)
        mixed_values = attention_weights @ self._values      # (token_count, 64)

        return mixed_values

class MultiHeadAttention:
    """Let every head study the same sentence, then combine what they found."""

    def __init__(self, block, head_count):
        self._qkv_projection = block.qkv_projection
        self._output_projection = block.attention_output_projection
        self._head_count = head_count

    def calculate(self, token_representations):
        """Run all attention heads and merge their results."""

        token_count = token_representations.shape[0]

        # Step 1 — Give each head its own Query, Key, and Value matrices.
        heads = self._create_heads(token_representations)

        # Step 2 — Build the "no peeking at the future" mask once for all heads.
        causal_mask = self._build_causal_mask(token_count, token_representations.dtype)

        # Step 3 — Run every head on the same sentence.
        # Unlike MultiHeadAttention itself (which is built once at load time),
        # SingleHeadAttention objects are created here on every prediction because
        # their Q, K, and V matrices are activations — they are derived from the
        # current prompt and change with every new input.
        head_outputs = []

        for head in heads:
            head_output = head.calculate(causal_mask)
            head_outputs.append(head_output)

        # Step 4 — Glue head outputs back together and apply the final projection.
        return self._combine_heads(head_outputs)

    def _create_heads(self, token_representations):
        """Hand every head its own Query, Key and Value to work with."""

        token_count = token_representations.shape[0]
        embedding_size = token_representations.shape[1]  # 768 for GPT-2 small
        head_size = embedding_size // self._head_count  # 768 / 12 = 64

        # Step 1 — One linear layer produces Query, Key, and Value together.
        # The weight matrix triples the width: 768 → 2304 = 3 × 768.
        projected = linear(
            token_representations,
            weight=self._qkv_projection.weight,
            bias=self._qkv_projection.bias,
        )                                                    # (token_count, 2304)

        # Step 2 — Split the big vector into three separate parts: Q, K, and V.
        # np.split(array, 3, axis=-1) cuts the columns into 3 equal groups.
        queries, keys, values = np.split(projected, 3, axis=-1)
        # queries, keys, values each: (token_count, 768)

        # Step 3 — Split each part across heads.
        # We go from one matrix of 768 numbers per token to 12 matrices of 64.
        # np.split returns a plain Python list of arrays.
        queries_per_head = np.split(queries, self._head_count, axis=-1)
        keys_per_head    = np.split(keys,    self._head_count, axis=-1)
        values_per_head  = np.split(values,  self._head_count, axis=-1)
        # each list: 12 elements of shape (token_count, 64)

        # Step 4 — Package one (Q, K, V) triple for each head.
        heads = []

        for head_index in range(self._head_count):
            head_queries = queries_per_head[head_index]
            head_keys    = keys_per_head[head_index]
            head_values  = values_per_head[head_index]
            heads.append(SingleHeadAttention(head_queries, head_keys, head_values))

        return heads

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
        allowed_positions  = np.tri(token_count, dtype=dtype) # (token_count, token_count)
        forbidden_positions = 1 - allowed_positions
        mask_penalty = -1e10

        return forbidden_positions * mask_penalty

    def _combine_heads(self, head_outputs):
        """Glue head outputs side by side, then let the model mix what each head found.

        np.hstack stacks arrays horizontally (column-wise) — the inverse of the
        np.split that created one matrix per head in _create_heads.
        After concatenation the output projection (a learned linear layer) mixes
        information across heads, so the model can blend what the different heads
        noticed into one coherent representation.
        """

        concatenated = np.hstack(head_outputs)               # (token_count, 768)

        return linear(
            concatenated,
            weight=self._output_projection.weight,
            bias=self._output_projection.bias,
        )                                                     # (token_count, 768)
