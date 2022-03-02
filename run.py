from experiments_framework import run_optimizer_experiment, get_avg_error, get_median_error, get_avg_iter, \
    get_median_iter, get_avg_norm_error, get_median_norm_error, evaluate_optimizer_results
from gradient_descent import normalize_loss
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor
from setup import optimizers_dict, loss_fn_dict, grad_fn_dict, tensors_kits
from store_load_weights import load_network_weights
import numpy as np


def print_aggregated_results_for_opt(opt_name, results_list, errors_dict):
    print(f"Summarized results for optimizer {opt_name}")
    print("Average Normalized Error:", get_avg_norm_error(results_list))
    print("Median Normalized Error:", get_median_norm_error(results_list))
    print("Average Normalized Clipping Error:", errors_dict['avg_norm_clip'])
    print("Average Normalized Rounding Error:", errors_dict['avg_norm_round'])
    print("Average Number of Iterations:", get_avg_iter(results_list))
    print("Median Number of Iterations::", get_median_iter(results_list))
    print("*iterations for self-implemented optimizers indicates the iteration number in which "
          "the best result was found, not the total number of iterations")
    print("")


if __name__ == "__main__":
    n_bits = 8
    # init_param_fn = lambda x: np.asarray([np.max(np.abs(x))])
    init_param_fn = lambda x: np.asarray([np.min(x), np.max(x)])

    # loaded_weights = load_network_weights(model_name='mobilenetv2',
    #                                       layers=tensors_kits['mobilenetv2_conv'])
    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=tensors_kits['mobilenetv2_depthwise'])
    weights_list = [weights for weights in loaded_weights.values()]
    per_channel = True

    # loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)
    # grad_fn = lambda t, x: grad_fn_dict['Threshold_MSE'](t, x, n_bits)
    loss_fn = lambda mm, x: loss_fn_dict['Min_Max_MSE'](mm, x, n_bits)
    grad_fn = lambda mm, x: grad_fn_dict['Min_Max_MSE'](mm, x, n_bits)

    for opt_name, opt_fn in optimizers_dict.items():
        optimizer = lambda init_param, x: opt_fn(init_param, x, loss_fn, grad_fn)

        results_list = run_optimizer_experiment(opt=optimizer,
                                                get_init_param=init_param_fn,
                                                weights_list=weights_list,
                                                per_channel=per_channel,
                                                opt_name=opt_name)

        errors_dict = evaluate_optimizer_results(results_list, n_bits)

        print(f"----- Optimizer {opt_name} results -----")
        for res in results_list:
            if not per_channel:
                print("Norm Loss:", res['norm_loss'], "Param:", res['param'])

        print_aggregated_results_for_opt(opt_name, results_list, errors_dict)