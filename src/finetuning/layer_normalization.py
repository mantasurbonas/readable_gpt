"""Layer normalization that can pass blame back."""

import numpy as np

from .adjustable_weights import AdjustableWeights
from .require_calculated import require_calculated


class LayerNormalization:
    """Same math as `utils.math_utils.layer_norm`, plus the memory needed for blaming.

    Every token's numbers are shifted and stretched to average 0 and spread 1,
    then multiplied by a learned scale and shifted by a learned offset.
    """

    EPSILON = 1e-5

    def __init__(self, layer_norm_parameters, checkpoint_name):
        self._scale = AdjustableWeights(layer_norm_parameters.scale, checkpoint_name + "/g")
        self._offset = AdjustableWeights(layer_norm_parameters.offset, checkpoint_name + "/b")
        self._remembered_standardized = None
        self._remembered_inverse_spread = None

    def calculate(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # "How many spreads away from the average is each number?" - 1 / spread, kept for later.
        inverse_spread = 1.0 / np.sqrt(variance + self.EPSILON)
        standardized = (x - mean) * inverse_spread

        self._remembered_standardized = standardized
        self._remembered_inverse_spread = inverse_spread

        return self._scale.values * standardized + self._offset.values

    def pass_blame_back(self, blame_for_output):
        """Share the blame for the output between the scale, the offset and the input.

        The scale and the offset are shared by all tokens, so they receive the
        blame of all rows added up.

        The input is trickier: every number influenced the average and the spread
        of its own row, and so influenced all the other numbers of that row. The
        formula below removes the parts of the blame that this "shift and stretch"
        could never produce, which is why it takes more than one line.
        """
        standardized = require_calculated(self._remembered_standardized, "LayerNormalization")
        inverse_spread = self._remembered_inverse_spread
        number_count = standardized.shape[-1]

        self._offset.add_blame(blame_for_output.sum(axis=0))
        self._scale.add_blame((blame_for_output * standardized).sum(axis=0))

        blame_for_standardized = blame_for_output * self._scale.values

        total_blame = blame_for_standardized.sum(axis=-1, keepdims=True)
        total_blame_weighted_by_position = (
            blame_for_standardized * standardized
        ).sum(axis=-1, keepdims=True)

        blame_for_input = (inverse_spread / number_count) * (
            number_count * blame_for_standardized
            - total_blame
            - standardized * total_blame_weighted_by_position
        )
        return blame_for_input

    def adjustable_weights(self):
        return [self._scale, self._offset]
