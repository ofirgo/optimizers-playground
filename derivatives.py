from typing import Callable

import numpy as np

from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor


def quantization_derivative_min_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/da (min level in quantization range)
    vec_der = np.vectorize(lambda _x: derivative_min_level(_x, a=a, b=b, n_bits=n_bits, s=(b - a) / (2 ** n_bits - 1)))
    return vec_der(x)


def derivative_min_level(x: float, a: float, b: float, n_bits: int, s: float):
    clip_elem = (x - a) / s
    if clip_elem <= 0:
        return 1
    elif clip_elem >= 2 ** n_bits - 1:
        return 0
    else:
        return ((x - a) / (b - a)) - (np.round(clip_elem) / (2 ** n_bits - 1))


def quantization_derivative_max_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/db (max level in quantization range)
    vec_der = np.vectorize(lambda _x: derivative_max_level(_x, a=a, b=b, n_bits=n_bits, s=(b - a) / (2 ** n_bits - 1)))
    return vec_der(x)


def derivative_max_level(x: float, a: float, b: float, n_bits: int, s: float):
    clip_elem = (x - a) / s
    if clip_elem <= 0:
        return 0
    elif clip_elem >= 2 ** n_bits - 1:
        return 1
    else:
        return (np.round(clip_elem) / (2 ** n_bits - 1)) - ((x - a) / (b - a))


def quantization_derivative_threshold(x: np.ndarray, t: float, n_bits: int):
    # calculates dQ/dt (quantization threshold)
    # only for signed quantization
    vec_der = np.vectorize(lambda _x: derivative_threshold(_x, t=t, n_bits=n_bits, s=(2 * t) / (2 ** n_bits)))
    return vec_der(x)


def derivative_threshold(x: float, t: float, n_bits: int, s: float):
    clip_elem = (x + t) / s
    if clip_elem <= 0:
        return -1
    elif clip_elem >= 2 ** n_bits - 1:
        return 0
    else:
        return -1 + (np.round(clip_elem) / (2 ** (n_bits - 1))) - x / t


def min_max_derivative(float_tensor: np.ndarray, a: float, b: float, n_bits: int, loss_fn_derivative: Callable):
    dQ_da = loss_fn_derivative(x=float_tensor,
                               q=uniform_quantize_tensor(float_tensor, range_min=a, range_max=b, n_bits=n_bits),
                               dQ=quantization_derivative_min_level(float_tensor, a=a, b=b, n_bits=n_bits))

    dQ_db = loss_fn_derivative(x=float_tensor,
                               q=uniform_quantize_tensor(float_tensor, range_min=a, range_max=b, n_bits=n_bits),
                               dQ=quantization_derivative_max_level(float_tensor, a=a, b=b, n_bits=n_bits))

    return np.asarray([dQ_da, dQ_db])


def mse_derivative(x: np.ndarray, q: np.ndarray, dQ: np.ndarray):
    n = len(x)
    return (-1 / n) * np.sum((x - q) * dQ)

