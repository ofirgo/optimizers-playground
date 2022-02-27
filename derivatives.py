import numpy as np


def quantization_derivative_min_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/da (min level in quantization range)
    # TODO: vectorize

    dQ = []
    for i in range(len(x)):
        x_i = x[i]
        if x_i <= 0:
            d = 1
        elif x_i >= 2 ** n_bits - 1:
            d = 2 - (1 / 2 ** (n_bits - 1))
        else:
            s = (b - a) / (2 ** (n_bits - 1))
            d = 1 - (1 / (2 ** n_bits - 1) * np.round((x_i - a) / s)) + (x_i - b) / (b - a)
        dQ.append(d)
    return dQ


def quantization_derivative_max_level(x: np.ndarray, a: float, b: float, n_bits: int):
    # calculates dQ/db (max level in quantization range)
    # TODO: vectorize

    dQ = []
    for i in range(len(x)):
        x_i = x[i]
        if x_i <= 0:
            d = 0
        elif x_i >= 2 ** n_bits - 1:
            d = 2 - (1 / 2 ** (n_bits - 1))
        else:
            s = (b - a) / (2 ** (n_bits - 1))
            d = (1 / (2 ** n_bits - 1) * np.round((x_i - a) / s)) + (a - x_i) / (b - a)
        dQ.append(d)
    return dQ


def quantization_derivative_threshold(x: np.ndarray, t: float, n_bits: int):
    # calculates dQ/dt (quantization threshold)
    # only for signed quantization
    # TODO: vectorize

    dQ = []
    for i in range(len(x)):
        x_i = x[i]
        if x_i <= -t:
            d = -1
        elif x_i >= 2 * t * (1 - 1 / (2 ** n_bits)):
            d = 2 - (1 / 2 ** (n_bits - 1))
        else:
            s = t / (2 ** (n_bits - 1))
            d = (1 / (2 ** n_bits - 1)) * np.round((x_i / s)) + - x_i
        dQ.append(d)
    return dQ


def mse_derivative(x: np.ndarray, q: np.ndarray, dQ: np.ndarray):
    n = len(x)
    return (-1 / n) * np.sum((x - q) * dQ)

