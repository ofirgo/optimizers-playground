import itertools
from operator import itemgetter

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from scipy import stats

from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    uniform_quantize_tensor, fix_range_to_include_zero
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from store_load_weights import load_network_weights


def plot_loss(loos_res):
    plt.plot(loos_res)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    plt.cla()


def plot_range(ranges):
    plt.plot(ranges)
    plt.xlabel("Iteration")
    plt.ylabel("Bounds")
    plt.show()
    plt.cla()


def iterative_fixed_range_search(init_param: np.ndarray, x: np.ndarray, loss_fn: Callable, n_intervals: int = 20,
                                 n_iter: int = 10, alpha: float = 0.6, beta: float = 1.2,
                                 tolerance: float = 1e-09, draw: bool = False, verbose: bool = False):
    scaler = np.array([alpha, beta])
    curr_param = init_param
    best = {"param": None, "loss": np.inf}

    # for drawing
    all_loss_res = []

    for n in range(n_iter):
        prev_best_loss = best['loss']
        curr_res = search_fixed_range_intervals(curr_param * scaler, x, loss_fn, n_intervals)
        curr_param = curr_res['param']
        all_loss_res.append(curr_res['loss'])
        best = min(best, curr_res, key=itemgetter('loss'))

        if verbose:
            print("Curr Loss:", curr_res['loss'])
            print("Best param:", best['param'])

        iters_loss_diff = prev_best_loss - curr_res['loss']
        if 0 < iters_loss_diff < tolerance:
            # improvement in last step is very small, therefore - finishing the search
            break

    if draw:
        plot_loss(all_loss_res)

    return best


def iterative_decreasing_range_search(init_param: np.ndarray, x: np.ndarray, loss_fn: Callable, n_intervals: int = 20,
                                      n_iter: int = 30, alpha: float = 0.6, beta: float = 1.2,
                                      factor: Tuple = (1.02, 0.98), freq: int = 10,
                                      tolerance: float = 1e-09, draw: bool = False, verbose: bool = False):
    scaler = np.array([alpha, beta])
    curr_param = init_param
    best = {"param": None, "loss": np.inf}

    # for drawing
    all_loss_res = []
    all_ranges = []

    for n in range(n_iter):
        prev_best_loss = best['loss']
        curr_res = search_fixed_range_intervals(curr_param * scaler, x, loss_fn, n_intervals)
        curr_param = curr_res['param']

        all_loss_res.append(curr_res['loss'])
        all_ranges.append(curr_param * scaler)

        best = min(best, curr_res, key=itemgetter('loss'))

        if verbose:
            print("Curr Loss:", curr_res['loss'])
            print("Best param:", best['param'])

        iters_loss_diff = prev_best_loss - curr_res['loss']
        if 0 < iters_loss_diff < tolerance:
            # improvement in last step is very small, therefore - finishing the search
            break

        # increase scaler to make range bounds narrower in next iteration
        if n % freq == 0:
            scaler *= factor
            # prevent min bound from exceeding max bound
            scaler = np.array([min(scaler[0], 0.97), max(scaler[1], 1.03)])

    if draw:
        plot_loss(all_loss_res)
        plot_range(all_ranges)

    return best


def iterative_dynamic_range_search(init_range: np.ndarray, x: np.ndarray, scalers: np.ndarray,
                                   loss_fn: Callable, n_iter: int = 10, tolerance: float = 1e-09,
                                   random_step: bool = False, freq: float = 10,
                                   draw: bool = False, verbose: bool = False):
    curr_range_bounds = init_range
    best = {"param": None, "loss": np.inf}

    # for drawing
    all_loss_res = []

    for n in range(n_iter):
        prev_best_loss = best['loss']
        curr_res = search_dynamic_range(base_range=curr_range_bounds, scalers=scalers, x=x, loss_fn=loss_fn)
        curr_range_bounds = curr_res['param']
        # curr_range_bounds = fix_range_to_include_zero(curr_range_bounds[0], curr_range_bounds[1], 8, False, 0)
        all_loss_res.append(curr_res['loss'])
        best = min(best, curr_res, key=itemgetter('loss'))

        if verbose:
            print("Curr Loss:", curr_res['loss'])
            print("Best param:", best['param'])

        iters_loss_diff = prev_best_loss - curr_res['loss']
        if 0 < iters_loss_diff < tolerance:
            # improvement in last step is very small, therefore - finishing the search
            break

        # change base range a bit by random step
        if random_step and n % freq == 0:
            mu = 1.0
            sigma = 0.2
            # (lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)
            step_size = stats.truncnorm((0.75 - mu) / sigma, (1.25 - mu) / sigma, loc=mu, scale=sigma)
            step_size = step_size.rvs(2)
            # step_size = np.random.normal(loc=1.0, scale=0.2, size=2)
            curr_range_bounds *= step_size


    if draw:
        plot_loss(all_loss_res)

    return best


def search_fixed_range_intervals(range_bounds: np.ndarray, x: np.ndarray, loss_fn: Callable, n_intervals: int = 20):
    intervals = np.linspace(start=range_bounds[0], stop=range_bounds[1], num=n_intervals, dtype=float)
    losses = list(map(lambda t: loss_fn(t, x), intervals))
    return {"param": np.array([intervals[np.argmin(losses)]]), "loss": np.min(losses)}


# def search_dynamic_range(base_range: np.ndarray, x: np.ndarray, alpha: np.ndarray, beta: np.ndarray, loss_fn: Callable):
#     ranges_min = base_range[0] * alpha
#     ranges_max = base_range[1] * beta
#     ranges = list(np.dstack((ranges_min, ranges_max)))[0]
#     losses = list(map(lambda r: loss_fn(r, x), ranges))
#     return {"param": ranges[np.argmin(losses)], "loss": np.min(losses)}

def search_dynamic_range(base_range: np.ndarray, x: np.ndarray, scalers: np.ndarray, loss_fn: Callable):
    ranges = base_range * scalers
    losses = list(map(lambda r: loss_fn(r, x), ranges))
    return {"param": ranges[np.argmin(losses)], "loss": np.min(losses)}


if __name__ == "__main__":
    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=['block_3_depthwise'])
    weights_tensor = loaded_weights['block_3_depthwise']['weights']
    print(weights_tensor.shape)
    weights_tensor = weights_tensor.flatten()

    n_bits = 8
    loss_fn = lambda t, float_tensor: compute_mse(float_tensor,
                                                  quantize_tensor(float_tensor, t, n_bits=n_bits, signed=True))
    max_tensor = np.max(np.abs(weights_tensor))

    # res = iterative_fixed_range_search(init_param=max_tensor,
    #                                    x=weights_tensor, loss_fn=loss_fn, n_intervals=50,
    #                                    n_iter=300, alpha=0.6, beta=1.2, tolerance=1e-08, draw=True, verbose=False)
    # print(res)

    # res = iterative_decreasing_range_search(init_param=max_tensor,
    #                                         x=weights_tensor, loss_fn=loss_fn, n_intervals=100,
    #                                         n_iter=300, alpha=0.5, beta=1.3, tolerance=1e-11,
    #                                         factor=(1.02, 0.98), freq=10,
    #                                         draw=True, verbose=False)
    # print(res)

    loss_fn = lambda r, float_tensor: compute_mse(float_tensor,
                                                  uniform_quantize_tensor(float_tensor, r[0], r[1], n_bits=n_bits))
    init_range = np.array([np.min(weights_tensor), np.max(weights_tensor)])
    print(init_range)
    alpha = np.linspace(0.7, 1.3, 10)
    beta = np.linspace(0.7, 1.3, 10)

    # To take 1-by-1 from alpha-beta use: scalers = list(np.dstack((alpha, beta)))[0] instead of product
    scalers = np.asarray(list(itertools.product(alpha, beta)))
    print(scalers.shape)
    res = iterative_dynamic_range_search(init_range=init_range, x=weights_tensor, scalers=scalers,
                                         loss_fn=loss_fn, n_iter=300, tolerance=1e-11,
                                         random_step=True, freq=20,
                                         draw=True, verbose=False)


    print(res)
