"""Pure NumPy math primitives used by the GPT forward pass.

--- A quick NumPy primer ---

`@`  — matrix multiplication.
      If A has shape (m, n) and B has shape (n, p), then (A @ B) has shape (m, p).
      Each output cell is a dot product of one row from A with one column from B.

`.T` — transpose: flip rows and columns.
      A matrix of shape (m, n) becomes (n, m).

Broadcasting — when an operation involves arrays of different shapes, NumPy
      stretches the smaller array to match the larger one, without copying data.
      Example: adding a bias vector of shape (768,) to a matrix of shape
      (token_count, 768) adds the same bias row to every token row.

`axis=-1`  — means "along the last axis", i.e. independently for each row.
`keepdims=True` — after reducing along an axis, keep a size-1 dimension in that
      place so the result broadcasts back over the original array correctly.
"""

import numpy as np


def gaussian_error_linear_unit_activation(x):
    """Apply GPT-2's GELU activation: a smooth, probabilistic version of ReLU.

    Uses the tanh approximation from the original GPT-2 implementation:
        0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (
        1
        + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        )
    )


def softmax(x):
    """Convert raw scores into a probability distribution along the last axis.

    Each row is processed independently (axis=-1): the scores within a row are
    turned into non-negative numbers that sum to 1.0.

    Why subtract the row maximum first?
    np.exp() overflows to infinity for inputs much above ~700. Subtracting the
    row maximum before exponentiating prevents that. The result is identical
    because the constant cancels in the ratio: exp(a - c) / sum(exp(b - c))
    is the same as exp(a) / sum(exp(b)) for any constant c.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, scale, offset, epsilon=1e-5):
    """Normalize each token's 768 numbers to zero mean and unit variance, then rescale.

    Works on each token row independently (axis=-1, keepdims=True).

    After normalization, 'scale' multiplies every number and 'offset' shifts it.
    Both are learned vectors of length 768 — the model trains them to undo any
    harmful effect of normalization and to give each position the right range.

    epsilon is a tiny constant added to the variance to avoid division by zero.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return scale * normalized + offset


def normalize_layer(token_representations, layer_norm_parameters, epsilon=1e-5):
    """Apply layer normalization using a scale/offset parameter bundle."""
    return layer_norm(
        token_representations,
        scale=layer_norm_parameters.scale,
        offset=layer_norm_parameters.offset,
        epsilon=epsilon,
    )


def linear(x, weight, bias):
    """Multiply by a learned weight matrix and add a learned bias — one neural-network layer.

    x @ weight projects the input into a new space (e.g. 768 → 2304 numbers).
    + bias adds the same learned offset to every token row (broadcasting).
    """
    return x @ weight + bias
