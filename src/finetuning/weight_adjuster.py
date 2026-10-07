"""Nudge every adjustable weight against its blame."""

import numpy as np


class WeightAdjuster:
    """Moves each weight a little in the direction that reduces the blame (the Adam recipe).

    A plain "subtract blame * learning rate" is jumpy: one noisy example can fling
    a weight far away. Adam smooths this by remembering, per number:
      - the average blame so far          (which direction has it usually been?)
      - the average squared blame so far  (how large and how jumpy has it been?)
    A weight with large, jumpy blame takes smaller steps; a weight with small,
    steady blame takes bigger ones.

    The weights are changed IN PLACE, in the same arrays the inference GPT reads.
    The two averages live only in this object and are not saved to disk.
    """

    def __init__(
        self,
        adjustable_weights,
        learning_rate=5e-5,
        average_blame_memory=0.9,
        average_squared_blame_memory=0.999,
        tiny_number_against_dividing_by_zero=1e-8,
        longest_allowed_total_blame=1.0,
    ):
        self._adjustable_weights = adjustable_weights
        self._learning_rate = learning_rate
        self._average_blame_memory = average_blame_memory
        self._average_squared_blame_memory = average_squared_blame_memory
        self._tiny_number = tiny_number_against_dividing_by_zero
        self._longest_allowed_total_blame = longest_allowed_total_blame

        self._adjustments_made = 0
        self._average_blame = {}
        self._average_squared_blame = {}
        for weights in adjustable_weights:
            self._average_blame[weights.checkpoint_name] = np.zeros_like(weights.values)
            self._average_squared_blame[weights.checkpoint_name] = np.zeros_like(weights.values)

    def adjust_weights(self):
        """Use the blame currently stored in the weights to make one adjustment."""
        self._limit_total_blame()
        self._adjustments_made += 1

        # The averages start at zero, so early on they are too small.
        # These factors undo that shrinkage; they approach 1 as adjustments pile up.
        average_blame_correction = 1.0 - self._average_blame_memory ** self._adjustments_made
        average_squared_blame_correction = 1.0 - self._average_squared_blame_memory ** self._adjustments_made

        for weights in self._adjustable_weights:
            name = weights.checkpoint_name
            average_blame = self._average_blame[name]
            average_squared_blame = self._average_squared_blame[name]

            average_blame *= self._average_blame_memory
            average_blame += (1.0 - self._average_blame_memory) * weights.blame

            average_squared_blame *= self._average_squared_blame_memory
            average_squared_blame += (1.0 - self._average_squared_blame_memory) * (weights.blame * weights.blame)

            corrected_average_blame = average_blame / average_blame_correction
            corrected_average_squared_blame = average_squared_blame / average_squared_blame_correction

            step = self._learning_rate * corrected_average_blame / (
                np.sqrt(corrected_average_squared_blame) + self._tiny_number
            )
            weights.values -= step

    def _limit_total_blame(self):
        """If the blame of the whole model is huge, shrink all of it by the same factor.

        This protects against a single freak example wrecking the model.
        "Total blame" is the length of all blame numbers laid out in one long list.
        """
        if self._longest_allowed_total_blame is None:
            return

        squared_total = 0.0
        for weights in self._adjustable_weights:
            squared_total += float(np.sum(weights.blame * weights.blame))
        total_blame = np.sqrt(squared_total)

        if total_blame <= self._longest_allowed_total_blame or total_blame == 0.0:
            return

        shrink_factor = self._longest_allowed_total_blame / total_blame
        for weights in self._adjustable_weights:
            weights.blame *= shrink_factor
