from experiments_framework import run_optimizer_experiment
from setup import optimizers_dict, loss_fn_dict, grad_fn_dict
from store_load_weights import load_network_weights
import numpy as np

if __name__ == "__main__":
    n_bits = 8
    init_param_fn = lambda x: np.asarray([np.max(np.abs(x))])
    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=['block_2_depthwise'])
    weights_list = [weights for weights in loaded_weights.values()]

    loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)
    grad_fn = lambda t, x: grad_fn_dict['Threshold_MSE'](t, x, n_bits)

    for opt_name, opt_fn in optimizers_dict.items():
        optimizer = lambda init_param, x: opt_fn(init_param, x, loss_fn, grad_fn)

        results_list = run_optimizer_experiment(opt=optimizer,
                                                get_init_param=init_param_fn,
                                                weights_list=weights_list,
                                                per_channel=False)
        if "Scipy" in opt_name:
            "Scipy algorithm, loss value is not normalized!"
            print(opt_name, results_list[0].fun, results_list[0].x)
        else:
            print(opt_name, results_list[0]['norm_loss'], results_list[0]['param'])