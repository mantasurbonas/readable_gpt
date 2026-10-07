"""One neural-network layer (multiply by weights, add a bias) that can pass blame back."""

from utils.math_utils import linear

from .adjustable_weights import AdjustableWeights
from .require_calculated import require_calculated


class LinearLayer:
    """Same math as `utils.math_utils.linear`, but it remembers its input
    and owns its weights, so the training can blame them.

        output = input @ weight + bias
    """

    def __init__(self, linear_parameters, checkpoint_name):
        self._weight = AdjustableWeights(linear_parameters.weight, checkpoint_name + "/w")
        self._bias = AdjustableWeights(linear_parameters.bias, checkpoint_name + "/b")
        self._remembered_input = None

    def calculate(self, x):
        self._remembered_input = x
        return linear(x, weight=self._weight.values, bias=self._bias.values)

    def pass_blame_back(self, blame_for_output):
        """Share the blame for the output between the weights, the bias and the input.

        - The weight matrix is blamed by pairing every input number with the
          blame of every output number it helped to produce.
        - The bias was added to every row equally, so it receives the sum of all rows' blame.
        - The input is blamed by sending the output blame back through the same weights.
        """
        x = require_calculated(self._remembered_input, "LinearLayer")

        self._weight.add_blame(x.T @ blame_for_output)
        self._bias.add_blame(blame_for_output.sum(axis=0))

        blame_for_input = blame_for_output @ self._weight.values.T
        return blame_for_input

    def adjustable_weights(self):
        return [self._weight, self._bias]
