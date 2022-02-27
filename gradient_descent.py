from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from derivatives import mse_derivative, quantization_derivative_threshold, quantization_derivative_min_level, \
    quantization_derivative_max_level, min_man_derivative
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    uniform_quantize_tensor
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from store_load_weights import load_network_weights


def draw_gd(n_iter, loss_res, param_res):
    plt.plot(range(n_iter), loss_res)
    plt.xlabel("Iteration")
    # plt.xticks(range(n_iter), fontsize=6)
    plt.ylabel("Loss")
    plt.show()
    plt.cla()

    plt.plot(range(n_iter), param_res)
    plt.xlabel("Iteration")
    # plt.xticks(range(n_iter), fontsize=6)
    plt.ylabel("Parameter Value")
    plt.show()
    plt.cla()


def sgd(param: np.ndarray, x: np.ndarray, loss_fn: Callable, gradient: Callable, n_epochs: int = 10,
        learn_rate: float = 0.1, tolerance: float = 1e-06, batch_size: int = 1, seed=1, draw: bool = False):
    # TODO: verify batch size is smaller than length of input

    # random split to batches
    rng = np.random.default_rng(seed=seed)

    vector = param  # this is the parameter we are optimizing
    loss_res = []
    param_res = []
    best = {"param": vector, "loss": loss_fn(vector, x), "it": 0}

    real_n_epochs = n_epochs
    for i in range(n_epochs):
        print(f"### Epoch {i} ###")
        rng.shuffle(x)

        loss = np.inf  # dummy
        grad = 0  # dummy
        epoch_loss = []
        for batch_start in range(0, len(x), batch_size):
            batch = x[batch_start:batch_start + batch_size]
            loss = loss_fn(vector, batch)
            best = best if loss >= best['loss'] else {"param": vector, "loss": loss, "it": i}
            if loss <= tolerance:
                break

            epoch_loss.append(loss)
            grad = gradient(vector, batch)
            diff = -learn_rate * grad
            vector += diff

        # log at the end of each epoch
        param_res.append(vector)
        loss_res.append(sum(epoch_loss) / batch_size)

        print(f"Param = {vector}")
        print(f"Loss: {loss}")
        print(f"Gradient: {grad}")
        print()

        if loss <= tolerance:
            # TODO: add to res dict indication about the cause for termination
            real_n_epochs = i + 1  # for plotting, in case we finished before completing all epochs
            break  # out of epoch loop

    if draw:
        draw_gd(real_n_epochs, loss_res, param_res)

    return best


def gradient_descent(param: np.ndarray, x: np.ndarray, loss_fn: Callable, gradient: Callable, n_iter: int = 50,
                     learn_rate: float = 0.1, tolerance: float = 1e-06, draw: bool = False):
    vector = param  # this is the parameter we are optimizing
    loss_res = []
    param_res = []
    best = {"param": vector, "loss": loss_fn(vector, x), "it": 0}
    real_n_iter = n_iter
    for i in range(n_iter):
        print(f"### Iteration {i} ###")
        param_res.append(vector)
        loss = loss_fn(vector, x)
        loss_res.append(loss)
        best = best if loss >= best['loss'] else {"param": vector, "loss": loss, "it": i}
        if loss <= tolerance:
            # TODO: add to res dict indication about the cause for termination
            real_n_iter = i + 1  # for plotting, in case we finished before completing all iterations
            break

        grad = gradient(vector, x)
        diff = -learn_rate * grad
        vector += diff

        print(f"Param = {vector}")
        print(f"Loss: {loss}")
        print(f"Gradient: {grad}")
        print()

    if draw:
        draw_gd(real_n_iter, loss_res, param_res)

    return best


def threshold_gd_example(weights_tensor, n_bits):
    loss_fn = lambda t, float_tensor: compute_mse(float_tensor,
                                                  quantize_tensor(float_tensor, t, n_bits=n_bits, signed=True))
    grad_fn = lambda t, float_tensor: mse_derivative(x=float_tensor,
                                                     q=quantize_tensor(float_tensor, t, n_bits=n_bits, signed=True),
                                                     dQ=quantization_derivative_threshold(float_tensor, t, n_bits))
    init_param = np.max(np.abs(weights_tensor))
    res = gradient_descent(param=init_param.copy(),
                           x=weights_tensor.copy(),
                           loss_fn=loss_fn,
                           gradient=grad_fn,
                           n_iter=30,
                           learn_rate=0.08,
                           tolerance=1e-06,
                           draw=True)

    batch_res = sgd(param=init_param.copy(),
                    x=weights_tensor.copy(),
                    loss_fn=loss_fn,
                    gradient=grad_fn,
                    n_epochs=50,
                    learn_rate=0.001,
                    tolerance=1e-06,
                    batch_size=10,
                    seed=2,
                    draw=True)

    print(init_param)
    print(res, batch_res)


def min_max_gd_example(weights_tensor, n_bits):
    loss_fn = lambda min_max, float_tensor: compute_mse(float_tensor,
                                                        uniform_quantize_tensor(float_tensor, range_min=min_max[0],
                                                                                range_max=min_max[1], n_bits=n_bits))
    grad_fn = lambda min_max, float_tensor: min_man_derivative(float_tensor, a=min_max[0], b=min_max[1], n_bits=n_bits,
                                                               loss_fn=mse_derivative)

    init_param = np.asarray([np.min(weights_tensor), np.max(weights_tensor)])
    res = gradient_descent(param=init_param.copy(),
                           x=weights_tensor.copy(),
                           loss_fn=loss_fn,
                           gradient=grad_fn,
                           n_iter=50,
                           learn_rate=0.01,
                           tolerance=1e-06,
                           draw=True)

    batch_res = sgd(param=init_param.copy(),
                    x=weights_tensor.copy(),
                    loss_fn=loss_fn,
                    gradient=grad_fn,
                    n_epochs=50,
                    learn_rate=0.001,
                    tolerance=1e-06,
                    batch_size=10,
                    seed=2,
                    draw=True)

    print("Init: ", init_param)
    print(res, batch_res)


if __name__ == "__main__":
    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=['block_2_depthwise_BN'])
    weights_tensor = loaded_weights['block_2_depthwise_BN']
    print(weights_tensor.shape)

    threshold_gd_example(weights_tensor, 8)
    # min_max_gd_example(weights_tensor, 8)


