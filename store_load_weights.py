import pickle
from tensorflow_fw.network_archives.tf_network_dict import tf_network_dict

"""
This code is used to export wights tensors of networks for side debugging and research purposes.
It should allow investigating issues and new tools without the necessary to run the MCT.
!!! Currently, only implemented for tensorflow models !!!
"""

WEIGHTS_DB_PATH = "/data/projects/swat/users/ofirgo/weights_db"
FILENAME_PREFIX = "weights"


def get_network_weights(model_name):
    weights = dict()  # layer_name -> weights_tensor
    model = tf_network_dict.get(model_name).get_model()
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            weights[layer.name] = layer_weights[0]  # index 0 is weights, index 1 is bias
    return weights


def load_network_weights(model_name, layers):
    # layers list can't be empty!
    weights_dict = dict()  # layer_name -> weights_tensor
    for layer_name in layers:
        layer_weights = load_layer_weights(model_name, layer_name)
        weights_dict[layer_name] = layer_weights
    return weights_dict


def load_layer_weights(model_name, layer_name):
    filename = f"{WEIGHTS_DB_PATH}/{FILENAME_PREFIX}_{model_name}_{layer_name}"
    with open(filename, 'rb') as infile:
        layer_weights = pickle.load(infile)
        return layer_weights


def store_network_weights(weights_dict, model_name, layers=None):
    if not layers:
        # store weights of all layers
        for layer_name, weights in weights_dict.items():
            store_layer_weights(model_name, layer_name, weights)
    else:
        # store weights of specific layers
        for layer_name in layers:
            store_layer_weights(model_name, layer_name, weights_dict[layer_name])


def store_layer_weights(model_name, layer_name, weights):
    filename = f"{WEIGHTS_DB_PATH}/{FILENAME_PREFIX}_{model_name}_{layer_name}"
    with open(filename, 'wb') as outfile:
        pickle.dump(weights, outfile)


def create_network_weights_db(model_name, layers_names_cond):
    model_weights = get_network_weights(model_name)  # layer_name -> weights_tensor
    layers_to_store = list(filter(lambda l: layers_names_cond(l), model_weights.keys()))

    store_network_weights(model_weights, model_name, layers_to_store)


if __name__ == "__main__":
    # weights_dict = get_network_weights('mobilenetv2')
    # store_network_weights(weights_dict, 'mobilenetv2', ['block_14_depthwise_BN'])
    # load_layer_weights('mobilenetv2', 'block_14_depthwise_BN')
    # print(WEIGHTS['mobilenetv2']['block_14_depthwise_BN'].shape)

    # create_network_weights_db(model_name='mobilenetv2',
    #                           layers_names_cond=lambda name: "block" in name and "depthwise_BN" in name)

    loaded_weights = load_network_weights(model_name='mobilenetv2',
                                          layers=['block_7_depthwise_BN',
                                                  'block_9_depthwise_BN',
                                                  'block_14_depthwise_BN'])
    for l_name, weights in loaded_weights.items():
        print(l_name, weights.shape)