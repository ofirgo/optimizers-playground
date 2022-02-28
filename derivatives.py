from typing import Callable

import numpy as np

from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor


def quantization_derivative_min_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/da (min level in quantization range)
    # TODO: vectorize

    dQ = []
    s = (b - a) / (2 ** n_bits - 1)
    for i in range(len(x)):
        x_i = x[i]
        clip_elem = (x_i - a) / s
        if clip_elem <= 0:
            d = 1
        elif clip_elem >= 2 ** n_bits - 1:
            d = 0
        else:
            d = 1 - (np.round(clip_elem) / (2 ** n_bits - 1)) - ((b - x_i) / (b - a))
        dQ.append(d)
    return dQ


def quantization_derivative_max_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/db (max level in quantization range)
    # TODO: vectorize

    dQ = []
    s = (b - a) / (2 ** n_bits - 1)
    for i in range(len(x)):
        x_i = x[i]
        clip_elem = (x_i - a) / s
        if clip_elem <= 0:
            d = 0
        elif clip_elem >= 2 ** n_bits - 1:
            d = 1
        else:
            d = (np.round(clip_elem) / (2 ** n_bits - 1)) + ((a - x_i) / (b - a))
        dQ.append(d)
    return dQ


def quantization_derivative_threshold(x: np.ndarray, t: float, n_bits: int):
    # calculates dQ/dt (quantization threshold)
    # only for signed quantization
    # TODO: vectorize

    dQ = []
    s = (2 * t) / (2 ** n_bits)
    for i in range(len(x)):
        x_i = x[i]
        clip_elem = (x_i + t) / s
        if clip_elem <= 0:
            d = -1
        elif clip_elem >= 2 ** n_bits - 1:
            d = 0
        else:
            d = -1 + (np.round(clip_elem) / (2 ** (n_bits - 1))) - x_i / t
        dQ.append(d)
    return dQ


def min_max_derivative(float_tensor: np.ndarray, a: float, b: float, n_bits: int, loss_fn: Callable):
    dQ_da = loss_fn(x=float_tensor,
                     q=uniform_quantize_tensor(float_tensor, range_min=a, range_max=b, n_bits=n_bits),
                     dQ=quantization_derivative_min_level(float_tensor, a=a, b=b, n_bits=n_bits))

    dQ_db = loss_fn(x=float_tensor,
                    q=uniform_quantize_tensor(float_tensor, range_min=a, range_max=b, n_bits=n_bits),
                    dQ=quantization_derivative_max_level(float_tensor, a=a, b=b, n_bits=n_bits))

    return np.asarray([dQ_da, dQ_db])


def mse_derivative(x: np.ndarray, q: np.ndarray, dQ: np.ndarray):
    n = len(x)
    return (-1 / n) * np.sum((x - q) * dQ)

