"""The numbers that training is allowed to change."""

import numpy as np


class AdjustableWeights:
    """A group of numbers the training may change: a weight matrix, a bias, a scale...

    `values` is the very same NumPy array that the inference GPT reads. It is
    never copied, so every adjustment is immediately visible to inference.

    `blame` has the same shape as `values`. Each cell answers:
        "If this value were a little bigger, how much bigger would
         the model's loss be?"
    A positive blame means "this value is pushing the model in the wrong
    direction", so the weight adjuster moves it down. A negative blame moves it up.
    """

    def __init__(self, values, checkpoint_name):
        self.values = values
        self.checkpoint_name = checkpoint_name  # e.g. "model/h3/attn/c_attn/w"
        self.blame = np.zeros_like(values)

    def add_blame(self, blame):
        """Blame from several places adds up, so we never overwrite it."""
        self.blame += blame

    def add_blame_to_rows(self, row_numbers, blame_per_row):
        """Add blame to the listed rows. A row listed twice receives both shares."""
        # Plain `self.blame[row_numbers] += ...` would apply only one share to a repeated row.
        np.add.at(self.blame, row_numbers, blame_per_row)

    def add_blame_to_first_rows(self, row_count, blame_per_row):
        self.blame[:row_count] += blame_per_row

    def forget_blame(self):
        self.blame[...] = 0
