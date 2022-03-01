from experiments_framework import run_optimizer_experiment
from gradient_descent import normalize_loss
from setup import optimizers_dict, loss_fn_dict, grad_fn_dict, tensors_kits
from store_load_weights import load_network_weights
import numpy as np

if __name__ == "__main__":
    n_bits = 8
    # init_param_fn = lambda x: np.asarray([np.max(np.abs(x))])
    init_param_fn = lambda x: np.asarray([np.min(x), np.max(x)])

    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=tensors_kits['mobilenetv2_conv'])
    # loaded_weights = load_network_weights(model_name='mobilenetv2',
    #                                       layers=tensors_kits['mobilenetv2_depthwise'])
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

        for res in results_list:
            print(opt_name, res['norm_loss'], res['param'])