"""The attention part of a Transformer block, able to pass blame back.

This file mirrors gpt/multi_head_attention.py step by step. Every `calculate()`
here does the same math as its twin there, and additionally remembers what
`pass_blame_back()` needs. `pass_blame_back()` walks the same steps in reverse order.
"""

import numpy as np
from utils.math_utils import softmax

from .linear_layer import LinearLayer
from .require_calculated import require_calculated


class SingleHeadAttention:
    """One head: let every token collect information from the tokens it cares about.

    A head has NO adjustable weights of its own. It only reads its own 64 columns
    of each of the Query, Key and Value blocks that the shared projection of
    MultiHeadAttention produced. So it can blame those columns, but the weights
    that produced them belong to MultiHeadAttention.
    """

    def __init__(self, head_size, head_number, block_index, tracer):
        self._head_size = head_size
        self._head_number = head_number
        self._block_index = block_index
        self._tracer = tracer

        self._remembered_projection_shape = None
        self._remembered_projection_dtype = None
        self._remembered_queries = None
        self._remembered_keys = None
        self._remembered_values = None
        self._remembered_attention_weights = None

    def calculate(self, projected_q_k_v, token_count):
        queries, keys, values = self._slice_head_kqv_from_projection(projected_q_k_v)

        raw_relevance_scores = queries @ keys.T
        scaled_relevance_scores = raw_relevance_scores / np.sqrt(self._head_size)
        causal_mask = self._build_causal_mask(token_count, projected_q_k_v.dtype)
        masked_scores = scaled_relevance_scores + causal_mask
        attention_weights = softmax(masked_scores)
        mixed_values = attention_weights @ values

        self._remembered_projection_shape = projected_q_k_v.shape
        self._remembered_projection_dtype = projected_q_k_v.dtype
        self._remembered_queries = queries
        self._remembered_keys = keys
        self._remembered_values = values
        self._remembered_attention_weights = attention_weights

        return mixed_values

    def pass_blame_back(self, blame_for_mixed_values):
        """Walk the five steps of `calculate()` in reverse.

        Returns the blame for the whole wide projected row. Only this head's
        columns are non-zero.
        """
        attention_weights = require_calculated(self._remembered_attention_weights, "SingleHeadAttention")
        queries = self._remembered_queries
        keys = self._remembered_keys
        values = self._remembered_values

        # Step 5 reversed - mixed_values = attention_weights @ values
        blame_for_values = attention_weights.T @ blame_for_mixed_values
        blame_for_attention_weights = blame_for_mixed_values @ values.T

        # Step 4 reversed - softmax turned every row into percentages that sum to 1.
        # Blame that would break that "sum to 1" rule is removed:
        #   blame_for_score = weight * (blame_for_weight - sum of (blame_for_weight * weight))
        blame_already_shared = np.sum(
            blame_for_attention_weights * attention_weights,
            axis=-1,
            keepdims=True,
        )
        blame_for_masked_scores = attention_weights * (
            blame_for_attention_weights - blame_already_shared
        )

        # Step 3 reversed - the mask is a constant that was added, so it takes no blame.
        blame_for_scaled_scores = blame_for_masked_scores

        # Step 2 reversed - the scores were divided by the square root of the head size.
        blame_for_raw_scores = blame_for_scaled_scores / np.sqrt(self._head_size)

        # Step 1 reversed - raw_scores = queries @ keys.T
        blame_for_queries = blame_for_raw_scores @ keys
        blame_for_keys = blame_for_raw_scores.T @ queries

        return self._place_head_blame_into_projection(
            blame_for_queries,
            blame_for_keys,
            blame_for_values,
        )

    def adjustable_weights(self):
        return []

    def _slice_head_kqv_from_projection(self, projected_q_k_v):
        """Pick this head's Query, Key, and Value out of the wide 2304-number row."""
        all_queries, all_keys, all_values = np.split(
            projected_q_k_v,
            3,
            axis=1,
        )

        head_start = self._head_number * self._head_size
        head_end = head_start + self._head_size

        queries = all_queries[:, head_start:head_end]
        keys = all_keys[:, head_start:head_end]
        values = all_values[:, head_start:head_end]

        return queries, keys, values

    def _place_head_blame_into_projection(self, blame_for_queries, blame_for_keys, blame_for_values):
        """The reverse of `_slice_head_kqv_from_projection`.

        Start from an all-zero wide row:

            [ 768 Query columns | 768 Key columns | 768 Value columns ]

        and write this head's 64 columns into each of the three blocks.
        Columns that belong to the other heads stay zero.
        """
        blame_for_projection = np.zeros(
            require_calculated(self._remembered_projection_shape, "SingleHeadAttention"),
            dtype=self._remembered_projection_dtype,
        )
        block_width = blame_for_projection.shape[1] // 3

        head_start = self._head_number * self._head_size
        head_end = head_start + self._head_size

        blame_for_projection[:, head_start:head_end] = blame_for_queries
        blame_for_projection[:, block_width + head_start:block_width + head_end] = blame_for_keys
        blame_for_projection[:, 2 * block_width + head_start:2 * block_width + head_end] = blame_for_values

        return blame_for_projection

    @staticmethod
    def _build_causal_mask(token_count, dtype):
        """Build the "you may not look forward" mask (see gpt/multi_head_attention.py)."""
        allowed_positions = np.tri(token_count, dtype=dtype)
        forbidden_positions = 1 - allowed_positions
        mask_penalty = -1e10

        return forbidden_positions * mask_penalty


class MultiHeadAttention:
    """Let every head study the same sentence, then combine what they found.

    Owns the two projections that hold all of attention's learned weights.
    """

    def __init__(self, block, head_count, block_index, tracer):
        self._block_index = block_index
        self._tracer = tracer

        checkpoint_prefix = f"model/h{block_index}/attn"
        self._qkv_projection = LinearLayer(block.qkv_projection, checkpoint_prefix + "/c_attn")
        self._output_projection = LinearLayer(block.attention_output_projection, checkpoint_prefix + "/c_proj")

        head_size = block.qkv_projection.weight.shape[-1] // 3 // head_count

        self._heads = []
        for head_number in range(head_count):
            self._heads.append(
                SingleHeadAttention(head_size, head_number, block_index, tracer)
            )

    def calculate(self, token_representations):
        token_count = token_representations.shape[0]

        projected_q_k_v = self._qkv_projection.calculate(token_representations)

        head_outputs = []
        for head in self._heads:
            head_outputs.append(head.calculate(projected_q_k_v, token_count))

        return self._combine_heads(head_outputs)

    def pass_blame_back(self, blame_for_output):
        # Step 3 reversed - hand every head its own share of the blame.
        blame_for_concatenated = self._output_projection.pass_blame_back(blame_for_output)
        blame_for_head_outputs = self._split_blame_between_heads(blame_for_concatenated)

        # Step 2 reversed - every head blames its own columns of the wide projected row.
        # The heads read different columns, so adding their blame together keeps them all.
        blame_for_projection = 0
        for head, blame_for_head_output in zip(self._heads, blame_for_head_outputs):
            blame_for_projection = blame_for_projection + head.pass_blame_back(blame_for_head_output)

        # Step 1 reversed - the one big Query/Key/Value multiply.
        return self._qkv_projection.pass_blame_back(blame_for_projection)

    def adjustable_weights(self):
        return self._qkv_projection.adjustable_weights() + self._output_projection.adjustable_weights()

    def _combine_heads(self, head_outputs):
        concatenated = np.hstack(head_outputs)
        return self._output_projection.calculate(concatenated)

    def _split_blame_between_heads(self, blame_for_concatenated):
        """The reverse of `np.hstack`: cut the wide row back into one piece per head."""
        return np.split(blame_for_concatenated, len(self._heads), axis=1)
