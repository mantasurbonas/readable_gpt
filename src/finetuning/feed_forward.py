"""The per-token neural network that sits after attention in every block, able to pass blame back."""

from .activation import GaussianErrorLinearUnitActivation
from .linear_layer import LinearLayer


class FeedForward:
    """Each token thinks about what it collected, on its own.

    Same three steps as gpt/feed_forward.py: stretch, bend, squeeze.
    """

    def __init__(self, block, block_index, tracer):
        self._block_index = block_index
        self._tracer = tracer

        checkpoint_prefix = f"model/h{block_index}/mlp"
        self._input_projection = LinearLayer(block.feed_forward_input_projection, checkpoint_prefix + "/c_fc")
        self._activation = GaussianErrorLinearUnitActivation()
        self._output_projection = LinearLayer(block.feed_forward_output_projection, checkpoint_prefix + "/c_proj")

    def calculate(self, token_representations):
        expanded = self._input_projection.calculate(token_representations)
        activated = self._activation.calculate(expanded)
        result = self._output_projection.calculate(activated)

        return result

    def pass_blame_back(self, blame_for_output):
        blame_for_activated = self._output_projection.pass_blame_back(blame_for_output)
        blame_for_expanded = self._activation.pass_blame_back(blame_for_activated)
        blame_for_input = self._input_projection.pass_blame_back(blame_for_expanded)

        return blame_for_input

    def adjustable_weights(self):
        return (
            self._input_projection.adjustable_weights()
            + self._activation.adjustable_weights()
            + self._output_projection.adjustable_weights()
        )
