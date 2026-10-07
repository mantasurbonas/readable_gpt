"""The bending step of the feed-forward network, with the ability to pass blame back."""

import numpy as np

from utils.math_utils import gaussian_error_linear_unit_activation

from .require_calculated import require_calculated


class GaussianErrorLinearUnitActivation:
    """Same math as `gaussian_error_linear_unit_activation`, remembering its input.

    It has no weights of its own, so there is nothing for it to adjust.
    """

    CUBIC_COEFFICIENT = 0.044715

    def __init__(self):
        self._remembered_input = None

    def calculate(self, x):
        self._remembered_input = x
        return gaussian_error_linear_unit_activation(x)

    def pass_blame_back(self, blame_for_output):
        """Each number is blamed on its own: blame times "how steeply the curve rises here".

        The curve is
            u = sqrt(2/pi) * (x + 0.044715 * x^3)
            y = 0.5 * x * (1 + tanh(u))
        and the steepness (slope) of it, at x, is worked out below.
        """
        x = require_calculated(self._remembered_input, "GaussianErrorLinearUnitActivation")

        curve_scale = np.sqrt(2.0 / np.pi)
        inner = curve_scale * (x + self.CUBIC_COEFFICIENT * x**3)
        tanh_of_inner = np.tanh(inner)
        inner_slope = curve_scale * (1.0 + 3.0 * self.CUBIC_COEFFICIENT * x**2)

        slope = (
            0.5 * (1.0 + tanh_of_inner)
            + 0.5 * x * (1.0 - tanh_of_inner**2) * inner_slope
        )
        return blame_for_output * slope

    def adjustable_weights(self):
        return []
