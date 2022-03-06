import numpy as np
from typing import Callable, Dict, List

from derivatives import mse_derivative, min_max_derivative
from optimizers.gradient_descent import gradient_descent, normalize_loss
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import \
    reshape_tensor_for_per_channel_search, uniform_quantize_tensor, fix_range_to_include_zero, calculate_delta
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from setup import min_max_mse_loss
from store_load_weights import load_network_weights


def run_optimizer_experiment(opt: Callable, get_init_param: Callable, weights_list: List[Dict], per_channel=False,
                             opt_name: str = None):
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
                if "Scipy" in opt_name:
                    res = adjust_scipy_results(res, channel_tensor)
                # add the actual optimized tensor to results for later evaluation
                res['tensor'] = channel_tensor
                results_list.append(res)
        else:
            init_param = get_init_param(tensor)
            res = opt(init_param.copy(),
                      tensor.copy().flatten())
            if "Scipy" in opt_name or "scipy" in opt_name:
                res = adjust_scipy_results(res, tensor)
            if "iterative_" in opt_name:
                res = adjust_iterative_results(res, tensor)
            # add the actual optimized tensor to results for later evaluation
            res['tensor'] = tensor
            results_list.append(res)

    return results_list


def adjust_scipy_results(res, x):
    return {"param": res.x, "loss": res.fun, "norm_loss": normalize_loss(res.fun, x), "it": res.nit,
            "status": res.status}


def adjust_iterative_results(res, x):
    return {"param": res['param'], "loss": res['loss'], "norm_loss": normalize_loss(res['loss'], x), "it": "NA",
            "status": "NA"}


def get_avg_error(results_list):
    errors_list = [res['loss'] for res in results_list]
    return np.average(errors_list)


def get_median_error(results_list):
    errors_list = [res['loss'] for res in results_list]
    return np.median(errors_list)


def get_avg_norm_error(results_list):
    errors_list = [res['norm_loss'] for res in results_list]
    return np.average(errors_list)


def get_median_norm_error(results_list):
    errors_list = [res['norm_loss'] for res in results_list]
    return np.median(errors_list)


def get_avg_iter(results_list):
    errors_list = [res['it'] for res in results_list]
    return np.average(errors_list)


def get_median_iter(results_list):
    errors_list = [res['it'] for res in results_list]
    return int(np.median(errors_list))


def evaluate_optimizer_results(results_list, n_bits):
    clips = []
    rounds = []
    for res in results_list:
        tensor = res['tensor']
        param = res['param']

        if len(param) == 1:
            # param is threshold, need to convert to min max range
            delta = calculate_delta(param, n_bits, signed=True)
            param = np.array([-param, param - delta])

        param = fix_range_to_include_zero(param[0], param[1], n_bits, False, 0)

        clip_err = compute_clipping_mse_error(tensor.flatten(), param, n_bits)
        round_err = compute_rounding_mse_error(tensor.flatten(), param, n_bits)
        clips.append(clip_err)
        rounds.append(round_err)

    return {"avg_norm_err": get_avg_norm_error(results_list), "med_norm_err": get_median_norm_error(results_list),
            "avg_norm_clip": np.average(clips), "avg_norm_round": np.average(rounds),
            "avg_loss": get_avg_error(results_list)}


def compute_clipping_mse_error(x, mm, n_bits):
    # returns the normalized clipping noise
    a, b = mm
    cond_idxs = np.where((x < a) | (x > b))[0]
    origin_data = x[cond_idxs]

    if len(cond_idxs) == 0:
        return 0

    err = min_max_mse_loss(np.array([a, b]), origin_data, n_bits)
    return normalize_loss(err, origin_data)


def compute_rounding_mse_error(channel_data, mm, n_bits):
    # returns the normalized rounding noise
    a, b = mm
    cond_idxs = np.where((channel_data > a) & (channel_data < b))[0]
    origin_data = channel_data[cond_idxs]

    if len(cond_idxs) == 0:
        return 0

    err = min_max_mse_loss(np.array([a, b]), origin_data, n_bits)
    return normalize_loss(err, origin_data)


if __name__ == "__main__":
    #####################
    ###### Example ######
    #####################
    n_bits = 8
    loss_fn = lambda min_max, float_tensor: compute_mse(float_tensor,
                                                        uniform_quantize_tensor(float_tensor, range_min=min_max[0],
                                                                                range_max=min_max[1], n_bits=n_bits))
    grad_fn = lambda min_max, float_tensor: min_max_derivative(float_tensor, a=min_max[0], b=min_max[1], n_bits=n_bits,
                                                               loss_fn_derivative=mse_derivative)

    init_param_fn = lambda x: np.asarray([np.min(x), np.max(x)])

    # optimizer = lambda x0, x: sgd(param=x0.copy(),
    #                               x=x.copy(),
    #                               loss_fn=loss_fn,
    #                               gradient=grad_fn,
    #                               n_epochs=30,
    #                               batch_size=10,
    #                               learn_rate=0.01,
    #                               tolerance=1e-06,
    #                               draw=False,
    #                               seed=2)

    optimizer = lambda x0, x: gradient_descent(param=x0.copy(),
                                               x=x.copy(),
                                               loss_fn=loss_fn,
                                               gradient=grad_fn,
                                               n_iter=50,
                                               learn_rate=0.01,
                                               tolerance=1e-06,
                                               draw=False)

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
