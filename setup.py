import itertools

from scipy import optimize
import numpy as np
from derivatives import mse_derivative, quantization_derivative_threshold, min_max_derivative
from optimizers.gradient_descent import gradient_descent
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    uniform_quantize_tensor
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from optimizers.iterative_opts import iterative_fixed_range_search, iterative_decreasing_range_search, \
    iterative_dynamic_range_search

"""
Loss Functions
"""
threshold_mse_loss = lambda t, x, n_bits: compute_mse(float_tensor=x,
                                                      fxp_tensor=quantize_tensor(x, t, n_bits=n_bits, signed=True))

min_max_mse_loss = lambda min_max, x, n_bits: compute_mse(float_tensor=x,
                                                          fxp_tensor=uniform_quantize_tensor(x,
                                                                                             range_min=min_max[0],
                                                                                             range_max=min_max[1],
                                                                                             n_bits=n_bits))

"""
Gradient Functions
"""
threshold_mse_grad = lambda t, x, n_bits: mse_derivative(x=x,
                                                         q=quantize_tensor(x, t, n_bits=n_bits, signed=True),
                                                         dQ=quantization_derivative_threshold(x, t, n_bits))

min_max_mse_grad = lambda min_max, x, n_bits: min_max_derivative(float_tensor=x, a=min_max[0], b=min_max[1],
                                                                 n_bits=n_bits, loss_fn_derivative=mse_derivative)

"""
Optimizers
"""
basic_gd = lambda init_param, x, loss_fn, grad_fn: gradient_descent(param=init_param.copy(),
                                                                    x=x.copy(),
                                                                    loss_fn=loss_fn,
                                                                    gradient=grad_fn,
                                                                    n_iter=50,
                                                                    learn_rate=1e-2,
                                                                    tolerance=1e-6,
                                                                    grad_norm=False,
                                                                    grad_noise=False,
                                                                    draw=False)

norm_grad_gd = lambda init_param, x, loss_fn, grad_fn: gradient_descent(param=init_param.copy(),
                                                                        x=x.copy(),
                                                                        loss_fn=loss_fn,
                                                                        gradient=grad_fn,
                                                                        n_iter=50,
                                                                        learn_rate=1e-2,
                                                                        tolerance=1e-6,
                                                                        grad_norm=True,
                                                                        grad_noise=False,
                                                                        draw=False)

noised_grad_gd = lambda init_param, x, loss_fn, grad_fn: gradient_descent(param=init_param.copy(),
                                                                          x=x.copy(),
                                                                          loss_fn=loss_fn,
                                                                          gradient=grad_fn,
                                                                          n_iter=50,
                                                                          learn_rate=1e-2,
                                                                          tolerance=1e-6,
                                                                          grad_norm=False,
                                                                          grad_noise=True,
                                                                          draw=False)

norm_noised_grad_gd = lambda init_param, x, loss_fn, grad_fn: gradient_descent(param=init_param.copy(),
                                                                               x=x.copy(),
                                                                               loss_fn=loss_fn,
                                                                               gradient=grad_fn,
                                                                               n_iter=50,
                                                                               learn_rate=1e-2,
                                                                               tolerance=1e-6,
                                                                               grad_norm=True,
                                                                               grad_noise=True,
                                                                               draw=False)

scipy_minimize_nelder_mead = lambda init_param, x, loss_fn, grad_fn: \
    optimize.minimize(fun=lambda param: loss_fn(param, x),
                      x0=init_param,
                      method='Nelder-Mead')

scipy_minimize_bfgs_no_grad = lambda init_param, x, loss_fn, grad_fn: \
    optimize.minimize(fun=lambda param: loss_fn(param, x),
                      x0=init_param,
                      method='BFGS')

scipy_minimize_bfgs_w_grad = lambda init_param, x, loss_fn, grad_fn: \
    optimize.minimize(fun=lambda param: loss_fn(param, x),
                      x0=init_param,
                      method='BFGS',
                      jac=lambda param: grad_fn(param, x))

threshold_optimizers_set = [
    {"name": "iterative_fixed_range_1",
     "opt": lambda t, x, loss_fn: iterative_fixed_range_search(init_param=t, x=x, loss_fn=loss_fn, n_intervals=50,
                                                               n_iter=100, alpha=0.7, beta=1.1, tolerance=1e-11),
     "config": {"n_intervals": 50, "n_iter": 100, "alpha": 0.7, "beta": 1.1}},

    {"name": "iterative_fixed_range_2",
     "opt": lambda t, x, loss_fn: iterative_fixed_range_search(init_param=t, x=x, loss_fn=loss_fn, n_intervals=100,
                                                               n_iter=100, alpha=0.6, beta=1.2, tolerance=1e-11),
     "config": {"n_intervals": 100, "n_iter": 100, "alpha": 0.6, "beta": 1.2}},

    {"name": "iterative_decreasing_range_1",
     "opt": lambda t, x, loss_fn: iterative_decreasing_range_search(init_param=t, x=x, loss_fn=loss_fn, n_intervals=100,
                                                                    n_iter=100, alpha=0.65, beta=1.25,
                                                                    tolerance=1e-11, factor=(1.015, 0.99), freq=10),
     "config": [{"n_intervals": 100, "n_iter": 100, "alpha": 0.65, "beta": 1.25, "factor": (1.015, 0.99), "freq": 10}]},

    {"name": "iterative_decreasing_range_2",
     "opt": lambda t, x, loss_fn: iterative_decreasing_range_search(init_param=t, x=x, loss_fn=loss_fn, n_intervals=100,
                                                                    n_iter=100, alpha=0.5, beta=1.3,
                                                                    tolerance=1e-11, factor=(1.02, 0.98), freq=10),
     "config": [{"n_intervals": 100, "n_iter": 100, "alpha": 0.5, "beta": 1.3, "factor": (1.02, 0.98), "freq": 10}]},

    {"name": "iterative_decreasing_range_3",
     "opt": lambda t, x, loss_fn: iterative_decreasing_range_search(init_param=t, x=x, loss_fn=loss_fn, n_intervals=100,
                                                                    n_iter=100, alpha=0.5, beta=1.3,
                                                                    tolerance=1e-11, factor=(1.02, 0.98), freq=20),
     "config": [{"n_intervals": 100, "n_iter": 100, "alpha": 0.5, "beta": 1.3, "factor": (1.02, 0.98), "freq": 20}]},

    # {"name": "norm_noised_gd",
    #  "opt": lambda t, x, loss_fn, grad_fn: norm_noised_grad_gd(init_param=t, x=x, loss_fn=loss_fn, grad_fn=grad_fn),
    #  "config": []},

    {"name": "scipy_nelder_mead",
     "opt": lambda t, x, loss_fn: scipy_minimize_nelder_mead(init_param=t, x=x, loss_fn=loss_fn, grad_fn=None),
     "config": []},
]

minmax_optimizers_set = [
    {"name": "iterative_fixed_range_1",
     "alpha": np.linspace(0.7, 1.3, 10),
     "beta": np.linspace(0.7, 1.3, 10),
     "sclaer_builder": lambda a, b: list(np.dstack((a, b)))[0],
     "opt": lambda mm, x, loss_fn, scalers: iterative_dynamic_range_search(init_range=mm, x=x, scalers=scalers,
                                                                           loss_fn=loss_fn, n_iter=200, tolerance=1e-11,
                                                                           random_step=False),
     "config": {"n_iter": 200, "scalers": "1-to-1", "alpha": (0.7, 1.3, 10), "beta": (0.7, 1.3, 10),
                "random_step": False}},

    # {"name": "iterative_fixed_range_2",
    #  "alpha": np.linspace(0.7, 1.3, 10),
    #  "beta": np.linspace(0.7, 1.3, 10),
    #  "sclaer_builder": lambda a, b: np.asarray(list(itertools.product(a, b))),
    #  "opt": lambda mm, x, loss_fn, scalers: iterative_dynamic_range_search(init_range=mm, x=x, scalers=scalers,
    #                                                                        loss_fn=loss_fn, n_iter=200, tolerance=1e-11,
    #                                                                        random_step=False),
    #  "config": {"n_iter": 200, "scalers": "product", "alpha": (0.7, 1.3, 10), "beta": (0.7, 1.3, 10)},
    #  "random_step": False},
    #
    # {"name": "iterative_fixed_range_3",
    #  "alpha": np.linspace(0.7, 1.3, 10),
    #  "beta": np.linspace(0.7, 1.3, 10),
    #  "sclaer_builder": lambda a, b: list(np.dstack((a, b)))[0],
    #  "opt": lambda mm, x, loss_fn, scalers: iterative_dynamic_range_search(init_range=mm, x=x, scalers=scalers,
    #                                                                        loss_fn=loss_fn, n_iter=200, tolerance=1e-11,
    #                                                                        random_step=True, freq=10),
    #  "config": {"n_iter": 200, "scalers": "1-to-1", "alpha": (0.7, 1.3, 10), "beta": (0.7, 1.3, 10),
    #             "random_step": True, "freq": 10}},
    #
    # {"name": "iterative_fixed_range_4",
    #  "alpha": np.linspace(0.7, 1.3, 10),
    #  "beta": np.linspace(0.7, 1.3, 10),
    #  "sclaer_builder": lambda a, b: np.asarray(list(itertools.product(a, b))),
    #  "opt": lambda mm, x, loss_fn, scalers: iterative_dynamic_range_search(init_range=mm, x=x, scalers=scalers,
    #                                                                        loss_fn=loss_fn, n_iter=200, tolerance=1e-11,
    #                                                                        random_step=True, freq=10),
    #  "config": {"n_iter": 200, "scalers": "product", "alpha": (0.7, 1.3, 10), "beta": (0.7, 1.3, 10),
    #             "random_step": True, "freq": 10}},
    #
    # {"name": "iterative_fixed_range_5",
    #  "alpha": np.linspace(0.55, 1.2, 10),
    #  "beta": np.linspace(0.55, 1.2, 10),
    #  "sclaer_builder": lambda a, b: np.asarray(list(itertools.product(a, b))),
    #  "opt": lambda mm, x, loss_fn, scalers: iterative_dynamic_range_search(init_range=mm, x=x, scalers=scalers,
    #                                                                        loss_fn=loss_fn, n_iter=200, tolerance=1e-11,
    #                                                                        random_step=True, freq=20),
    #  "config": {"n_iter": 200, "scalers": "product", "alpha": (0.55, 1.2, 10), "beta": (0.55, 1.2, 10),
    #             "random_step": True, "freq": 20}},
    #
    # # scalers is dummy
    # {"name": "scipy_nelder_mead",
    #  "opt": lambda mm, x, loss_fn, scalers: scipy_minimize_nelder_mead(init_param=mm, x=x, loss_fn=loss_fn, grad_fn=None),
    #  "config": []},
]

"""
Components Dicts
"""
optimizers_dict = {
    "Basic_GD": basic_gd,
    "Norm_Grad_GD": norm_grad_gd,
    "Noised_Grad_GD": noised_grad_gd,
    "Norm_Noised_Grad_GD": norm_noised_grad_gd,
    "Scipy_Nelder_Mead": scipy_minimize_nelder_mead,
    "Scipy_BFGS_No_Grad": scipy_minimize_bfgs_no_grad,
    "Scipy_BFGS_W_Grad": scipy_minimize_bfgs_w_grad,
}

loss_fn_dict = {
    "Threshold_MSE": threshold_mse_loss,
    "Min_Max_MSE": min_max_mse_loss,
}

grad_fn_dict = {
    "Threshold_MSE": threshold_mse_grad,
    "Min_Max_MSE": min_max_mse_grad,
}

"""
Tensors Kits
"""
tensors_kits = {
    # "mobilenetv2_predictions": ['predictions'],  # large channels (1280x1000)
    "mobilenetv2_conv": ['Conv1', 'Conv_1'],  # two convolution layers (small and large)
    # "mobilenetv2_depthwise": ['block_1_depthwise', 'block_4_depthwise', 'block_7_depthwise', 'block_9_depthwise',
    #                           'block_14_depthwise'],  # multiple small channels (3x3)
    # "mobilenetv2_project": ['block_2_project', 'block_5_project', 'block_8_project', 'block_10_project',
    #                         'block_15_project'],  # multiple medium channels
    # "mobilenet_conv_dw": ['conv_dw_1', 'conv_dw_2', 'conv_dw_5', 'conv_dw_6', 'conv_dw_11', 'conv_dw_12'],
    # "mobilenet_conv_pw": ['conv_dw_4', 'conv_dw_7', 'conv_dw_8', 'conv_dw_9', 'conv_dw_10', 'conv_dw_13'],
    # "resnet50_conv2": ['conv2_block1_1_conv', 'conv2_block1_2_conv', 'conv2_block1_3_conv', 'conv2_block2_1_conv',
    #                    'conv2_block2_2_conv', 'conv2_block2_3_conv', 'conv2_block3_1_conv'],
    # "resnet50_conv4": ['conv4_block1_0_conv', 'conv4_block2_1_conv', 'conv4_block2_3_conv', 'conv4_block3_2_conv',
    #                    'conv4_block5_3_bn', 'conv4_block6_2_conv', 'conv4_block6_3_conv'],
}
