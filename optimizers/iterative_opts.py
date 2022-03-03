from operator import itemgetter

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt

from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from store_load_weights import load_network_weights


def plot_loss(loos_res):
    plt.plot(loos_res)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
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

        # increase scaler to make range bounds narrower in next iteration
        if n % freq == 0:
            scaler *= factor
            # prevent min bound from exceeding max bound
            scaler = np.array([min(scaler[0], 0.99), max(scaler[1], 1.01)])

    if draw:
        plot_loss(all_loss_res)

    return best


def search_fixed_range_intervals(range_bounds: np.ndarray, x: np.ndarray, loss_fn: Callable, n_intervals: int = 20):
    intervals = np.linspace(start=range_bounds[0], stop=range_bounds[1], num=n_intervals, dtype=float)
    losses = list(map(lambda t: loss_fn(t, x), intervals))
    return {"param": intervals[np.argmin(losses)], "loss": np.min(losses)}


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
    print(max_tensor)

    # res = iterative_fixed_range_search(init_param=max_tensor,
    #                                    x=weights_tensor, loss_fn=loss_fn, n_intervals=50,
    #                                    n_iter=300, alpha=0.6, beta=1.2, tolerance=1e-08, draw=True, verbose=False)
    # print(res)

    res = iterative_decreasing_range_search(init_param=max_tensor,
                                            x=weights_tensor, loss_fn=loss_fn, n_intervals=100,
                                            n_iter=300, alpha=0.6, beta=1.2, tolerance=1e-20,
                                            factor=(1.02, 0.98), freq=10,
                                            draw=True, verbose=False)
    print(res)
