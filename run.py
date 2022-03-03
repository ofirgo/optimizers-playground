from experiments_framework import run_optimizer_experiment, get_avg_iter, \
    get_median_iter, get_avg_norm_error, get_median_norm_error, evaluate_optimizer_results
from optimizers.iterative_opts import iterative_fixed_range_search, iterative_decreasing_range_search
from setup import optimizers_dict, loss_fn_dict, grad_fn_dict, tensors_kits, threshold_optimizers_set, \
    minmax_optimizers_set
from store_load_weights import load_network_weights
import numpy as np
import pandas as pd
from time import gmtime, strftime
import os


RESULTS_DIR = "/data/projects/swat/users/ofirgo/optimizers_results"

def print_aggregated_results_for_opt(opt_name, results_list, errors_dict):
    print(f"Summarized results for optimizer {opt_name}")
    print("Average Normalized Error:", errors_dict['avg_norm_err'])
    print("Median Normalized Error:", errors_dict['med_norm_err'])
    print("Average Normalized Clipping Error:", errors_dict['avg_norm_clip'])
    print("Average Normalized Rounding Error:", errors_dict['avg_norm_round'])
    # print("Average Number of Iterations:", get_avg_iter(results_list))
    # print("Median Number of Iterations::", get_median_iter(results_list))
    # print("*iterations for self-implemented optimizers indicates the iteration number in which "
    #       "the best result was found, not the total number of iterations")
    print("")


def run_gradient_base_opts():
    n_bits = 8
    init_param_fn = lambda x: np.asarray([np.max(np.abs(x))])
    # init_param_fn = lambda x: np.asarray([np.min(x), np.max(x)])

    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=tensors_kits['mobilenetv2_conv'])
    # loaded_weights = load_network_weights(model_name='mobilenetv2',
    #                                       layers=tensors_kits['mobilenetv2_depthwise'])
    weights_list = [weights for weights in loaded_weights.values()]
    per_channel = False

    loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)
    grad_fn = lambda t, x: grad_fn_dict['Threshold_MSE'](t, x, n_bits)
    # loss_fn = lambda mm, x: loss_fn_dict['Min_Max_MSE'](mm, x, n_bits)
    # grad_fn = lambda mm, x: grad_fn_dict['Min_Max_MSE'](mm, x, n_bits)

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


def run_iterative_threshold_opts():
    n_bits = 8
    init_param_fn = lambda x: np.array([np.max(np.abs(x))])

    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=tensors_kits['mobilenetv2_conv'])
    weights_list = [weights for weights in loaded_weights.values()]
    per_channel = False
    loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)

    # optimizer = lambda init_param, x: iterative_fixed_range_search(init_param=init_param,
    #                                                                x=x, loss_fn=loss_fn, n_intervals=50,
    #                                                                n_iter=300, alpha=0.6, beta=1.2, tolerance=1e-11,
    #                                                                draw=False, verbose=False)

    optimizer = lambda init_param, x: iterative_decreasing_range_search(init_param=init_param,
                                                                        x=x, loss_fn=loss_fn, n_intervals=100,
                                                                        n_iter=300, alpha=0.5, beta=1.3,
                                                                        tolerance=1e-11,
                                                                        factor=(1.02, 0.98), freq=10,
                                                                        draw=False, verbose=False)

    results_list = run_optimizer_experiment(opt=optimizer,
                                            get_init_param=init_param_fn,
                                            weights_list=weights_list,
                                            per_channel=per_channel,
                                            opt_name="iterative_fixed_range")

    print(results_list[0]['loss'], results_list[0]['param'])

    # res = iterative_decreasing_range_search(init_param=max_tensor,
    #                                         x=weights_tensor, loss_fn=loss_fn, n_intervals=100,
    #                                         n_iter=300, alpha=0.5, beta=1.3, tolerance=1e-11,
    #                                         factor=(1.02, 0.98), freq=10,
    #                                         draw=True, verbose=False)
    # print(res)


def threshold_selection_experiment():
    n_bits_test = [2, 3, 4, 5, 6, 7, 8]
    init_param_fn = lambda x: np.array([np.max(np.abs(x))])
    optimizers = threshold_optimizers_set
    test_kits = tensors_kits

    per_channel = False

    start_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_dir = f"{RESULTS_DIR}/threshold/{start_time}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    res_dicts = []
    for n_bits in n_bits_test:
        loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)
        for kit_name, kit_layer in test_kits.items():
            # Load weights of current test kit
            print(f"----- Running tests on kit {kit_name} with layers {kit_layer} -----")
            model_name = kit_name.split("_")[0]
            loaded_weights = load_network_weights(model_name=model_name, layers=kit_layer)
            weights_list = [weights for weights in loaded_weights.values()]

            for optimizer in optimizers:
                # Run optimizer test on layers in kit
                print(f"Running Optimizer {optimizer['name']}", f"with configuration: {optimizer['config']}")
                kit_results_list = run_optimizer_experiment(opt=lambda t, x: optimizer['opt'](t, x, loss_fn),
                                                            get_init_param=init_param_fn,
                                                            weights_list=weights_list,
                                                            per_channel=per_channel,
                                                            opt_name=optimizer['name'])

                errors_dict = evaluate_optimizer_results(kit_results_list, n_bits)
                print_aggregated_results_for_opt(optimizer['name'], kit_results_list, errors_dict)

                log_dict = {**optimizer, **errors_dict, 'kit': kit_name, 'n_bits': n_bits}
                res_dicts.append(log_dict)
    res_df = pd.DataFrame(res_dicts)
    res_df.to_csv(f"{output_dir}/all_results.csv")


def minmax_selection_experiment():
    n_bits_test = [2, 3, 4, 5, 6, 7, 8]
    init_param_fn = lambda x: np.array([np.min(x), np.max(x)])
    optimizers = minmax_optimizers_set
    test_kits = tensors_kits

    per_channel = False

    start_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_dir = f"{RESULTS_DIR}/minmax/{start_time}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    res_dicts = []
    for n_bits in n_bits_test:
        loss_fn = lambda t, x: loss_fn_dict['Threshold_MSE'](t, x, n_bits)
        for kit_name, kit_layer in test_kits.items():
            # Load weights of current test kit
            print(f"----- Running tests on kit {kit_name} with layers {kit_layer} -----")
            model_name = kit_name.split("_")[0]
            loaded_weights = load_network_weights(model_name=model_name, layers=kit_layer)
            weights_list = [weights for weights in loaded_weights.values()]

            for optimizer in optimizers:
                # Run optimizer test on layers in kit
                print(f"Running Optimizer {optimizer['name']}", f"with configuration: {optimizer['config']}")
                kit_results_list = run_optimizer_experiment(opt=lambda mm, x: optimizer['opt'](mm, x, loss_fn,
                                                                                               scalers=optimizer['sclaer_builder'](optimizer['alpha'],
                                                                                                                                   optimizer['beta'])),
                                                            get_init_param=init_param_fn,
                                                            weights_list=weights_list,
                                                            per_channel=per_channel,
                                                            opt_name=optimizer['name'])

                errors_dict = evaluate_optimizer_results(kit_results_list, n_bits)
                print_aggregated_results_for_opt(optimizer['name'], kit_results_list, errors_dict)

                log_dict = {**optimizer, **errors_dict, 'kit': kit_name, 'n_bits': n_bits}
                res_dicts.append(log_dict)
    res_df = pd.DataFrame(res_dicts)
    res_df.to_csv(f"{output_dir}/all_results.csv")


def plot_threshold_results():
    output_dir = f"{RESULTS_DIR}/threshold/2022-03-03_14:12:44"
    df = pd.from_csv(f"{output_dir}/all_results.csv")

    kits_list = df['kit'].unique()
    bits_list = df['n_bits'].unique()
    for kit in kits_list:
        kit_rows_df = df.loc[df['kit'] == kit]
        for bits in bits_list:
            bits_rows_df = kit_rows_df.loc[kit_rows_df['n_bits'] == bits]
            bits_rows_df.plot.bar(w=bits_rows_df['name'], y=bits_rows_df['avg_norm_err'])
            bits_rows_df.plot.bar(w=bits_rows_df['name'], y=bits_rows_df['avg_norm_clip'])
            bits_rows_df.plot.bar(w=bits_rows_df['name'], y=bits_rows_df['avg_norm_round'])

    for bits in bits_list:
        bits_rows_df = df.loc[df['n_bits'] == bits]
        agg_per_opt_df = bits_rows_df.groupby('name', as_index=False)
        avg_err_per_opt_df = agg_per_opt_df['avg_norm_err'].mean()
        avg_clip_per_opt_df = agg_per_opt_df['avg_norm_clip'].mean()
        avg_round_per_opt_df = agg_per_opt_df['avg_norm_clip'].mean()



if __name__ == "__main__":
    # threshold_selection_experiment()
    minmax_selection_experiment()