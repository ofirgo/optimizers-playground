
from scipy import optimize

from derivatives import mse_derivative, quantization_derivative_threshold, min_max_derivative
from optimizers.gradient_descent import gradient_descent
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor, \
    uniform_quantize_tensor
from model_compression_toolkit.common.similarity_analyzer import compute_mse


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
    "mobilenetv2_predictions": ['predictions'],  # large channels (1280x1000)
    "mobilenetv2_conv": ['Conv1', 'Conv_1'],  # two convolution layers (small and large)
    "mobilenetv2_depthwise": ['block_1_depthwise', 'block_4_depthwise', 'block_7_depthwise', 'block_9_depthwise', 'block_14_depthwise'],  # multiple small channels (3x3)
    "mobilenetv2_project": ['block_2_project', 'block_5_project', 'block_8_project', 'block_10_project', 'block_15_project'],  # multiple medium channels
}
