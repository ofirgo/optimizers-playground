import numpy as np
from typing import Callable, Dict, List

from derivatives import mse_derivative, min_max_derivative
from gradient_descent import gradient_descent, sgd
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import \
    reshape_tensor_for_per_channel_search, uniform_quantize_tensor
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from store_load_weights import load_network_weights


def run_optimizer_experiment(opt: Callable, get_init_param: Callable, weights_list: List[Dict], per_channel=False):
    # opt should be a wrapper that already contains all optimizer parameters, except init value and tensor data
    # get_init_param should except a tensor and return the init value for the tensors parameters optimization

    results_list = []
    for weights_obj in weights_list:
        # Run optimizer for each weight's tensor in list
        tensor = weights_obj['weights']
        channel_axis = None if not per_channel else weights_obj['channel_axis']

        if per_channel and channel_axis:
            tensor_r = reshape_tensor_for_per_channel_search(tensor, channel_axis)
            for j in range(tensor_r.shape[0]):  # iterate all channels of the tensor.
                channel_tensor = tensor_r[j, :]
                init_param = get_init_param(channel_tensor)
                res = opt(init_param.copy(),
                          channel_tensor.copy())
                results_list.append(res)
        else:
            init_param = get_init_param(tensor)
            res = opt(init_param.copy(),
                      tensor.copy().flatten())
            results_list.append(res)

    return results_list


def get_avg_error(results_list):
    errors_list = [res['loss'] for res in results_list]
    return np.average(errors_list)


def get_median_error(results_list):
    errors_list = [res['loss'] for res in results_list]
    return np.median(errors_list)


def get_avg_iter(results_list):
    errors_list = [res['it'] for res in results_list]
    return np.average(errors_list)


def get_median_iter(results_list):
    errors_list = [res['it'] for res in results_list]
    return int(np.median(errors_list))


if __name__ == "__main__":
    #####################
    ###### Example ######
    #####################
    n_bits = 8
    loss_fn = lambda min_max, float_tensor: compute_mse(float_tensor,
                                                        uniform_quantize_tensor(float_tensor, range_min=min_max[0],
                                                                                range_max=min_max[1], n_bits=n_bits))
    grad_fn = lambda min_max, float_tensor: min_max_derivative(float_tensor, a=min_max[0], b=min_max[1], n_bits=n_bits,
                                                               loss_fn=mse_derivative)

    init_param_fn = lambda x: np.asarray([np.min(x), np.max(x)])

    optimizer = lambda x0, x: sgd(param=x0.copy(),
                                  x=x.copy(),
                                  loss_fn=loss_fn,
                                  gradient=grad_fn,
                                  n_epochs=30,
                                  batch_size=10,
                                  learn_rate=0.01,
                                  tolerance=1e-06,
                                  draw=False,
                                  seed=2)

    # optimizer = lambda x0, x: gradient_descent(param=x0.copy(),
    #                                            x=x.copy(),
    #                                            loss_fn=loss_fn,
    #                                            gradient=grad_fn,
    #                                            n_iter=50,
    #                                            learn_rate=0.01,
    #                                            tolerance=1e-06,
    #                                            draw=False)

    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=['block_2_depthwise_BN', 'block_8_project_BN', 'block_14_depthwise'])
    # weights_list = [{'weights': weights, 'channel_axis': channel_axis}
    #                 for weights, channel_axis in loaded_weights.values()]
    weights_list = [weights for weights in loaded_weights.values()]

    results_list = run_optimizer_experiment(opt=optimizer,
                                            get_init_param=init_param_fn,
                                            weights_list=weights_list,
                                            per_channel=False)

    print(results_list)
    print("Average Error:", get_avg_error(results_list))
    print("Median Error:", get_median_error(results_list))
    print("Average Number of Iterations:", get_avg_iter(results_list))
    print("Median Number of Iterations::", get_median_iter(results_list))