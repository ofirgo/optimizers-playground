import pickle
from tensorflow_fw.network_archives.tf_network_dict import tf_network_dict
from model_compression_toolkit.keras.default_framework_info import DEFAULT_CHANNEL_AXIS_DICT

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
        channel_axis = DEFAULT_CHANNEL_AXIS_DICT.get(type(layer))[1]
        if len(layer_weights) > 0:
            weights[layer.name] = {'weights': layer_weights[0], 'channel_axis': channel_axis}  # index 0 in layer_weights is weights, index 1 is bias
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
    channels_filename = f"{WEIGHTS_DB_PATH}/channels"
    with open(filename, 'rb') as infile, open(channels_filename, 'r') as channels_file:
        # get layer's channel axis
        channels = [line.rstrip('\n') for line in channels_file]
        channels = list(filter(lambda c: f"{model_name}_{layer_name}" in c, channels))
        if len(channels) == 0:
            print(f"Layer {layer_name} channel axis is not recorded, considering no channel axis")
            channel_axis = None
        else:
            channel_axis = channels[0].split(" ")[1]
            channel_axis = None if channel_axis == "None" else int(channel_axis)
            if not channel_axis:
                print(f"Layer {layer_name} channel axis is None - can't optimize per channel")

        layer_weights = pickle.load(infile)
        return {'weights': layer_weights, 'channel_axis': channel_axis}


def store_network_weights(weights_dict, model_name, layers=None):
    if not layers:
        # store weights of all layers
        for layer_name, weights in weights_dict.items():
            # weights is a dict with weights tensor and channel axis
            store_layer_weights(model_name, layer_name, weights)
    else:
        # store weights of specific layers
        for layer_name in layers:
            store_layer_weights(model_name, layer_name, weights_dict[layer_name])


def store_layer_weights(model_name, layer_name, weights):
    # weights is a dict with weights tensor and channel axis !!!
    filename = f"{WEIGHTS_DB_PATH}/{FILENAME_PREFIX}_{model_name}_{layer_name}"
    channels_filename = f"{WEIGHTS_DB_PATH}/channels"
    with open(filename, 'wb') as outfile, open(channels_filename, 'a') as channels_file:
        channels_file.write(f"{model_name}_{layer_name} {weights['channel_axis']}\n")
        pickle.dump(weights['weights'], outfile)


def create_network_weights_db(model_name, layers_names_cond):
    model_weights = get_network_weights(model_name)  # layer_name -> weights_tensor
    layers_to_store = list(filter(lambda l: layers_names_cond(l), model_weights.keys()))

    store_network_weights(model_weights, model_name, layers_to_store)


def create_network_all_weights_db(model_name):
    model_weights = get_network_weights(model_name)  # layer_name -> weights_tensor

    store_network_weights(model_weights, model_name, None)


if __name__ == "__main__":
    ####
    # Single layer store and load example
    ####
    # weights_dict = get_network_weights('mobilenetv2')
    # print(weights_dict['block_14_depthwise']['weights'].shape, weights_dict['block_14_depthwise']['channel_axis'])
    # print(weights_dict['block_14_depthwise_BN']['weights'].shape, weights_dict['block_14_depthwise_BN']['channel_axis'])
    # store_network_weights(weights_dict, 'mobilenetv2', ['block_14_depthwise'])
    # store_network_weights(weights_dict, 'mobilenetv2', ['block_14_depthwise_BN'])
    # loaded_weights = load_network_weights('mobilenetv2', ['block_14_depthwise', 'block_14_depthwise_BN'])
    # print(loaded_weights['block_14_depthwise']['weights'].shape, loaded_weights['block_14_depthwise']['channel_axis'])
    # print(loaded_weights['block_14_depthwise_BN']['weights'].shape, loaded_weights['block_14_depthwise_BN']['channel_axis'])

    ####
    # Store layers by pattern
    ####
    # create_network_weights_db(model_name='mobilenetv2',
    #                           layers_names_cond=lambda name: "block" in name and "depthwise" in name)

    ####
    # Store all layers
    ####
    model_name = 'resnet50'
    create_network_all_weights_db(model_name)

    ####
    # Load multiple layers
    ####
    # loaded_weights = load_network_weights(model_name=model_name,
    #                      layers=['block_8_project_BN', 'block_8_project', 'block_8_expand_BN', 'block_8_expand',
    #                              'block_8_depthwise_BN', 'block_8_depthwise'])
    # for l_name, weights in loaded_weights.items():
    #     print(l_name, weights['weights'].shape, weights['channel_axis'])

    ####
    # Print all layer names and shapes
    ####
    loaded_weights = get_network_weights(model_name)
    for name, weights in loaded_weights.items():
        print(name, weights['weights'].shape, weights['channel_axis'])