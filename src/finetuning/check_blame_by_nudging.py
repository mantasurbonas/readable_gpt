"""Check that every part passes blame back correctly, by nudging its numbers.

Run from the project root (not as a plain script, because this package contains a
file named gpt.py that must not hide the real `gpt` package):

    set PYTHONPATH=src
    .venv\\Scripts\\python.exe -m finetuning.check_blame_by_nudging

The idea, for one part:
  1. Calculate the part's output and ask it to pass some made-up blame back.
  2. Now check the answer the slow, obvious way: nudge ONE number up a tiny bit,
     calculate again, and see how much the result moved. Then nudge it down.
     The difference tells how much that number really deserves to be blamed.
  3. Repeat for every input number and every adjustable weight, and compare.

Nothing is loaded from disk. The parts are built from tiny random numbers.
"""

import sys

import numpy as np

from gpt.model_loader import (
    LayerNormParameters,
    LinearParameters,
    TransformerBlockParameters,
)
from utils.tracer import Tracer

from .activation import GaussianErrorLinearUnitActivation
from .feed_forward import FeedForward
from .trainable_gpt import TrainableGPT
from .layer_normalization import LayerNormalization
from .linear_layer import LinearLayer
from .multi_head_attention import MultiHeadAttention, SingleHeadAttention
from .next_token_loss_calculator import NextTokenLossCalculator
from .transformer_block import TransformerBlock

NUDGE_SIZE = 1e-5
ALLOWED_DIFFERENCE = 1e-5

TINY_MODEL_SIZE = 8
TINY_HEAD_COUNT = 2
TINY_VOCABULARY_SIZE = 11
TINY_CONTEXT_SIZE = 6
TINY_BLOCK_COUNT = 2


def main():
    np.random.seed(0)

    checks = [
        ("linear layer", check_linear_layer),
        ("GELU activation", check_activation),
        ("layer normalization", check_layer_normalization),
        ("next-token loss", check_next_token_loss),
        ("one attention head", check_single_head_attention),
        ("multi-head attention", check_multi_head_attention),
        ("feed forward", check_feed_forward),
        ("transformer block", check_transformer_block),
        ("whole GPT", check_whole_gpt),
    ]

    failed = 0
    for name, check in checks:
        try:
            check()
            print(f"ok   {name}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error}")

    if failed:
        print(f"{failed} check(s) failed.")
        return 1

    print("All checks passed.")
    return 0


def check_linear_layer():
    layer = LinearLayer(_random_linear_parameters(5, 6), "model/test")
    _check_part_by_nudging(layer.calculate, layer.pass_blame_back, layer.adjustable_weights(), _randn(4, 5))


def check_activation():
    activation = GaussianErrorLinearUnitActivation()
    _check_part_by_nudging(activation.calculate, activation.pass_blame_back, [], _randn(3, 7))


def check_layer_normalization():
    normalization = LayerNormalization(_random_layer_norm_parameters(8), "model/test")
    _check_part_by_nudging(
        normalization.calculate,
        normalization.pass_blame_back,
        normalization.adjustable_weights(),
        _randn(3, 8),
    )


def check_single_head_attention():
    head_size = 5
    token_count = 4
    head = SingleHeadAttention(head_size, head_number=1, block_index=0, tracer=Tracer())

    # Two heads side by side: head number 1 must only blame its own columns.
    projected_width = 3 * 2 * head_size
    _check_part_by_nudging(
        lambda projected: head.calculate(projected, token_count),
        head.pass_blame_back,
        [],
        _randn(token_count, projected_width),
    )


def check_multi_head_attention():
    block = _random_block_parameters()
    attention = MultiHeadAttention(block, TINY_HEAD_COUNT, block_index=0, tracer=Tracer())
    _check_part_by_nudging(
        attention.calculate,
        attention.pass_blame_back,
        attention.adjustable_weights(),
        _randn(4, TINY_MODEL_SIZE),
    )


def check_feed_forward():
    block = _random_block_parameters()
    feed_forward = FeedForward(block, block_index=0, tracer=Tracer())
    _check_part_by_nudging(
        feed_forward.calculate,
        feed_forward.pass_blame_back,
        feed_forward.adjustable_weights(),
        _randn(4, TINY_MODEL_SIZE),
    )


def check_transformer_block():
    block = TransformerBlock(_random_block_parameters(), TINY_HEAD_COUNT, block_index=0, tracer=Tracer())
    _check_part_by_nudging(
        block.calculate,
        block.pass_blame_back,
        block.adjustable_weights(),
        _randn(4, TINY_MODEL_SIZE),
    )


def check_next_token_loss():
    logits = _randn(3, 5)
    expected_token_ids = [0, 2, 4]
    loss_calculator = NextTokenLossCalculator()

    loss_calculator.calculate(logits, expected_token_ids)
    blame = loss_calculator.pass_blame_back()

    nudged_blame = np.zeros_like(logits)
    for cell in np.ndindex(logits.shape):
        nudged_up = logits.copy()
        nudged_down = logits.copy()
        nudged_up[cell] += NUDGE_SIZE
        nudged_down[cell] -= NUDGE_SIZE
        loss_up = loss_calculator.calculate(nudged_up, expected_token_ids)
        loss_down = loss_calculator.calculate(nudged_down, expected_token_ids)
        nudged_blame[cell] = (loss_up - loss_down) / (2 * NUDGE_SIZE)

    _assert_close("blame for logits", blame, nudged_blame)


def check_whole_gpt():
    gpt = _build_tiny_gpt()
    token_ids = [3, 1, 4, 1, 5]
    _check_part_by_nudging(
        lambda _unused: gpt.score_every_next_token(token_ids),
        gpt.pass_blame_back,
        gpt.adjustable_weights(),
        None,
        forget_blame=gpt.forget_blame,
    )


def _check_part_by_nudging(calculate, pass_blame_back, adjustable_weights, input_values, forget_blame=None):
    """Compare the blame a part reports with the blame found by nudging.

    To turn "the whole output moved" into one number, every output cell is
    weighted by a made-up blame, and the weighted cells are summed up.
    """
    output = calculate(input_values)
    made_up_blame = _randn(*output.shape)

    for weights in adjustable_weights:
        weights.forget_blame()
    if forget_blame:
        forget_blame()

    reported_blame_for_input = pass_blame_back(made_up_blame)
    reported_blame_for_weights = {w.checkpoint_name: w.blame.copy() for w in adjustable_weights}

    def weighted_output():
        return np.sum(calculate(input_values) * made_up_blame)

    if input_values is not None:
        nudged = _blame_found_by_nudging(input_values, weighted_output)
        _assert_close("blame for input", reported_blame_for_input, nudged)

    for weights in adjustable_weights:
        nudged = _blame_found_by_nudging(weights.values, weighted_output)
        _assert_close(
            f"blame for {weights.checkpoint_name}",
            reported_blame_for_weights[weights.checkpoint_name],
            nudged,
        )


def _blame_found_by_nudging(numbers, weighted_output):
    """Nudge each number up and down in place, restoring it afterwards."""
    found = np.zeros_like(numbers)
    for cell in np.ndindex(numbers.shape):
        original = numbers[cell]

        numbers[cell] = original + NUDGE_SIZE
        result_up = weighted_output()
        numbers[cell] = original - NUDGE_SIZE
        result_down = weighted_output()
        numbers[cell] = original

        found[cell] = (result_up - result_down) / (2 * NUDGE_SIZE)
    return found


def _build_tiny_gpt():
    return TrainableGPT(
        token_embeddings=_randn(TINY_VOCABULARY_SIZE, TINY_MODEL_SIZE),
        position_embeddings=_randn(TINY_CONTEXT_SIZE, TINY_MODEL_SIZE),
        blocks=[_random_block_parameters() for _ in range(TINY_BLOCK_COUNT)],
        final_layer_norm=_random_layer_norm_parameters(TINY_MODEL_SIZE),
        attention_head_count=TINY_HEAD_COUNT,
        context_size=TINY_CONTEXT_SIZE,
    )


def _random_block_parameters():
    model_size = TINY_MODEL_SIZE
    return TransformerBlockParameters(
        attention_norm=_random_layer_norm_parameters(model_size),
        qkv_projection=_random_linear_parameters(model_size, 3 * model_size),
        attention_output_projection=_random_linear_parameters(model_size, model_size),
        feed_forward_norm=_random_layer_norm_parameters(model_size),
        feed_forward_input_projection=_random_linear_parameters(model_size, 4 * model_size),
        feed_forward_output_projection=_random_linear_parameters(4 * model_size, model_size),
    )


def _random_linear_parameters(input_size, output_size):
    return LinearParameters(weight=_randn(input_size, output_size), bias=_randn(output_size))


def _random_layer_norm_parameters(size):
    return LayerNormParameters(scale=_randn(size), offset=_randn(size))


def _randn(*shape):
    return np.random.randn(*shape).astype(np.float64)


def _assert_close(name, reported, found_by_nudging):
    if reported.shape != found_by_nudging.shape:
        raise AssertionError(
            f"{name}: shape {reported.shape} differs from nudged shape {found_by_nudging.shape}"
        )
    biggest_difference = float(np.max(np.abs(reported - found_by_nudging)))
    if biggest_difference > ALLOWED_DIFFERENCE:
        raise AssertionError(
            f"{name}: biggest difference {biggest_difference:.3e} exceeds {ALLOWED_DIFFERENCE:.3e}"
        )


if __name__ == "__main__":
    sys.exit(main())
