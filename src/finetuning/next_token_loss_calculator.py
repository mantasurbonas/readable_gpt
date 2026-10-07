"""Measure how wrong the model was about the token that really came next."""

import numpy as np

from utils.math_utils import softmax

from .require_calculated import require_calculated


class NextTokenLossCalculator:
    """Turn the model's scores into one number: how wrong was it about the truth?

    1. softmax turns each position's scores into percentages over the vocabulary.
    2. For each position look at the percentage given to the token that REALLY came next.
    3. Loss = the average of  -log(that percentage).
       100% on the right token gives a loss of 0. The smaller the percentage,
       the bigger the loss.
    """

    TINY_NUMBER_AGAINST_LOG_OF_ZERO = 1e-12

    def __init__(self):
        self._remembered_percentages = None
        self._remembered_expected_token_ids = None

    def calculate(self, logits, expected_next_token_ids):
        """`logits` has shape (position_count, vocabulary_size): one row of scores per position."""
        if logits.ndim != 2:
            raise ValueError("logits must have shape (position_count, vocabulary_size).")
        if len(expected_next_token_ids) != logits.shape[0]:
            raise ValueError("Need exactly one expected token id per position.")

        percentages = softmax(logits)
        positions = np.arange(logits.shape[0])
        percentages_of_truth = percentages[positions, expected_next_token_ids]

        self._remembered_percentages = percentages
        self._remembered_expected_token_ids = expected_next_token_ids

        loss = -np.mean(np.log(percentages_of_truth + self.TINY_NUMBER_AGAINST_LOG_OF_ZERO))
        return float(loss)

    def pass_blame_back(self):
        """How much is every score to blame for the loss?

        Because the percentages come from softmax, the answer is simple:
          - every word is blamed by its own percentage ("you were given this much attention"),
          - the word that really came next is relieved by 1 ("but you deserved 100%"),
          - all divided by the number of positions, because the loss is an average.
        """
        percentages = require_calculated(self._remembered_percentages, "NextTokenLossCalculator")
        expected_token_ids = self._remembered_expected_token_ids
        position_count = percentages.shape[0]

        blame = percentages.copy()
        blame[np.arange(position_count), expected_token_ids] -= 1.0
        blame /= position_count
        return blame
