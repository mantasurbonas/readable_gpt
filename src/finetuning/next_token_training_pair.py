"""One piece of text prepared for next-token training."""


class NextTokenTrainingPair:
    """A text, cut into "what the model sees" and "what it should have guessed".

    Take the tokens   [The, cat, sat, on]. The model reads the first three
    and at each position guesses the following token:

        sees:      The   cat   sat
        should say: cat   sat   on

    So the expected tokens are the same list, moved one position to the left.
    One short text therefore gives many guesses to learn from.
    """

    def __init__(self, token_ids, max_length):
        if len(token_ids) < 2:
            raise ValueError("The text is shorter than two tokens, so there is nothing to predict.")

        self.original_token_count = len(token_ids)
        kept_token_ids = token_ids[:max_length]

        self.input_token_ids = kept_token_ids[:-1]
        self.expected_next_token_ids = kept_token_ids[1:]

    @property
    def was_trimmed(self):
        return self.original_token_count > len(self.input_token_ids) + 1

    @property
    def position_count(self):
        return len(self.input_token_ids)
