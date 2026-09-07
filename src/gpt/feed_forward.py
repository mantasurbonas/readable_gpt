"""The per-token neural network that sits after attention in every block."""

from utils.math_utils import gaussian_error_linear_unit_activation, linear


class FeedForward:
    """Each token thinks about what it collected, on its own.

    Attention gathers information from other tokens. However this part does NOT look
    at other tokens at all - every row is processed independently. It
    stretches each token's 768 numbers out to 3072, bends them with an
    activation function, then squeezes them back down to 768.
    """

    def __init__(self, block):
        self._input_projection = block.feed_forward_input_projection
        self._output_projection = block.feed_forward_output_projection

    def calculate(self, token_representations):
        # Stretch: 768 → 3072 numbers per token.
        expanded = linear(
            token_representations,
            weight=self._input_projection.weight,
            bias=self._input_projection.bias,
        )                                                        # (token_count, 3072)

        # Bend: apply the activation function element-wise.
        # This introduces non-linearity — without it the two linear layers would
        # collapse into one and lose expressive power.
        activated = gaussian_error_linear_unit_activation(expanded)  # (token_count, 3072)

        # Squeeze: 3072 → 768 numbers per token.
        result = linear(
            activated,
            weight=self._output_projection.weight,
            bias=self._output_projection.bias,
        )                                                        # (token_count, 768)

        return result
