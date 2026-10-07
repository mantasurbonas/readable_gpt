"""Teach the model with examples."""

from .next_token_loss_calculator import NextTokenLossCalculator
from .trainable_gpt import TrainableGPT
from .weight_adjuster import WeightAdjuster


class FineTuner:
    """Show the model a training pair, calculate its loss, share out the blame, adjust the weights.

    Works directly on the arrays of the given ModelLoader, so afterwards the
    inference GPT built from the same loader is the improved model.
    """

    def __init__(self, model_loader, learning_rate):
        self._model = TrainableGPT(
            token_embeddings=model_loader.token_embeddings,
            position_embeddings=model_loader.position_embeddings,
            blocks=model_loader.blocks,
            final_layer_norm=model_loader.final_layer_norm,
            attention_head_count=model_loader.attention_head_count,
            context_size=model_loader.context_size,
        )
        self._loss_calculator = NextTokenLossCalculator()
        self._weight_adjuster = WeightAdjuster(
            self._model.adjustable_weights(),
            learning_rate=learning_rate,
        )

    def train_on(self, training_pair):
        """Learn from one training pair. Returns the loss the model had BEFORE learning."""
        self._model.forget_blame()

        logits = self._model.score_every_next_token(training_pair.input_token_ids)
        loss = self._loss_calculator.calculate(logits, training_pair.expected_next_token_ids)

        self._model.pass_blame_back(self._loss_calculator.pass_blame_back())
        self._weight_adjuster.adjust_weights()

        return loss

    def adjustable_weights(self):
        return self._model.adjustable_weights()
