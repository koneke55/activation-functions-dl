from __future__ import annotations

from typing import Optional

import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    out = np.zeros_like(x, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + z[pos_mask])
    out[neg_mask] = z[neg_mask] / (1.0 + z[neg_mask])
    return out


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, negative_slope * x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))


def selu(x: np.ndarray) -> np.ndarray:
    # SELU constants from the original paper
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))


def softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.where(x > 0.0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


def gelu(x: np.ndarray, approximate: bool = True) -> np.ndarray:
    if approximate:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    # exact: 0.5 * x * (1 + erf(x / sqrt(2))) but avoids SciPy
    from math import erf, sqrt

    vec_erf = np.vectorize(erf)
    return 0.5 * x * (1.0 + vec_erf(x / np.sqrt(2.0)))


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return x * sigmoid(beta * x)


def mish(x: np.ndarray) -> np.ndarray:
    return x * np.tanh(softplus(x))


def softmax(x: np.ndarray, axis: Optional[int] = -1) -> np.ndarray:
    # subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip((x * 0.2) + 0.5, 0.0, 1.0)


