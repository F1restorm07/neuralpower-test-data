import json

def power_search(arch, epoch, stride, batch_size, dropout_keep_prob):
    """
        find the predicted power for the given model
        arch: the given architecture -> [conv_arch, reduction_arch]
        epoch: the current running epoch
        stride: the stride for the x side and y side
        batch_size: the batch size for the input tensor
        dropout_keep_prob: the dropout layer's probability to keep the data
    """
    arch_map = {
        "name" : f"arch_{epoch}",
        "layers": {
            "data": {
                "parents": [],
                "type": "Input",
                "tensor": [batch_size, 3, 32, 32] # as defined for the CIFAR dataset (3 channels, width and height of 32)
            },
            "stem": {
                "type": "Convolution",
                "parents": ["data"],
                "filter": [3, 3, 32, 20],
                "padding": "SAME",
                "strides": [1, 1, 1, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm",
            }
        }
    }
    nodes = 5 # default in NAO
    x_conv_cnt = 0
    x_pool_cnt = 0
    y_conv_cnt = 0
    y_pool_cnt = 0
    x_prev_layer = "stem"
    y_prev_layer = "stem"

    for i in range(nodes):
        # as defined in NAO
        x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]

        x_stride = stride if x_id in [0, 1] else 1
        if x_op == 0:
            x_conv_cnt += 1
            layer_name = f"x/conv_{x_conv_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Convolution",
                "parents": [x_prev_layer],
                "filter": [3, 3, 20, 20],
                "padding": "SAME",
                "strides": [1, x_stride, x_stride, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm"
            }
            x_prev_layer = layer_name # this layer is now the previous layer
        elif x_op == 1:
            x_conv_cnt += 1
            layer_name = f"x/conv_{x_conv_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Convolution",
                "parents": [x_prev_layer],
                "filter": [5, 5, 20, 20],
                "padding": "SAME",
                "strides": [1, x_stride, x_stride, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm"
            }
            x_prev_layer = layer_name
        elif x_op == 2:
            x_pool_cnt += 1
            layer_name = f"x/pool_{x_pool_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "AvgPool",
                "parents": [x_prev_layer],
                "ksize": [1, 3, 3, 1], # as defined in NAO
                "strides": [1, x_stride, x_stride, 1],
                "padding": "SAME"
            }
            x_prev_layer = layer_name
        elif x_op == 3:
            x_pool_cnt += 1
            layer_name = f"x/pool_{x_pool_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Pooling",
                "parents": [x_prev_layer],
                "ksize": [1, 3, 3, 1], # as defined in NAO
                "strides": [1, x_stride, x_stride, 1],
                "padding": "SAME"
            }
            x_prev_layer = layer_name

        y_stride = stride if y_id in [0, 1] else 1
        if y_op == 0:
            y_conv_cnt += 1
            layer_name = f"y/conv_{y_conv_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Convolution",
                "parents": [y_prev_layer],
                "filter": [3, 3, 20, 20],
                "padding": "SAME",
                "strides": [1, y_stride, y_stride, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm"
            }
            y_prev_layer = layer_name
        elif y_op == 1:
            y_conv_cnt += 1
            layer_name = f"y/conv_{y_conv_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Convolution",
                "parents": [y_prev_layer],
                "filter": [5, 5, 20, 20],
                "padding": "SAME",
                "strides": [1, y_stride, y_stride, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm"
            }
            y_prev_layer = layer_name
        elif y_op == 2:
            y_pool_cnt += 1
            layer_name = f"y/pool_{y_pool_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "AvgPool",
                "parents": [y_prev_layer],
                "ksize": [1, 3, 3, 1], # as defined in NAO
                "strides": [1, y_stride, y_stride, 1],
                "padding": "SAME"
            }
            y_prev_layer = layer_name
        elif y_op == 3:
            y_pool_cnt += 1
            layer_name = f"y/pool_{y_pool_cnt}"
            arch_map["layers"][layer_name] = {
                "type": "Pooling",
                "parents": [y_prev_layer],
                "ksize": [1, 3, 3, 1], # as defined in NAO
                "strides": [1, y_stride, y_stride, 1],
                "padding": "SAME"
            }
            y_prev_layer = layer_name

    # AdaptiveAvgPool2d
    arch_map["layers"]["global_pooling"] = {
        "type": "AvgPool",
        # "parents": [x_prev_layer, y_prev_layer],
        "parents": [x_prev_layer],
        "ksize": [1, 1, 1, 1],
        "strides": [1, 1, 1, 1],
        "padding": "VALID",
    }
    arch_map["layers"]["drop"] = {
        "type": "Dropout",
        "parents": ["global_pooling"],
        "dropout_keep_prob": 1 - dropout_keep_prob
    }
    # with open("exp/search_cifar10/arch.json") as f:
    with open("arch.json", "w+") as f:
        f.write(json.dumps(arch_map, indent=4))

    # operation map:
    # 0: separable convolution -> kernel size 3
    # --> pointwise convolution, depthwise convolution
    # ----> pointwise convolution:
    # 1: separable convolution -> kernel size 5
    # 2: average pool
    # 3: max pool
    # 4: factorized reduce (we don't care about this layer)

    # pooling:
    # padding: "SAME"
    # ksize: [1, 3, 3, 1]
    # stride: [1, _, _, 1]
    # type: {"Pooling", "AvgPool"}

    # convolution:
    # type: "Convolution"
    # filter: [{3, 5}, {3, 5}, 20, 20], <- [height, width, chan_in, chan_out]
    # padding: "SAME",
    # strides: [1, _, _, 1],
    # activation_fn: "relu",
    # normalizer_fn: "batch_norm"

    # variables:
    # channels: 20 (both width and height)

if __name__ == "__main__":
    power_search(
        [ # arch
            1, 3, 0, 0,
            2, 4, 2, 4,
            3, 0, 3, 0,
            3, 1, 3, 1,
            3, 4, 3, 3,
            1, 3, 0, 0,
            2, 3, 2, 0,
            3, 0, 3, 0,
            3, 1, 3, 0,
            3, 4, 3, 3,
        ],
        1, # epoch
        2, # stride
        64, # batch_size
        1.0 # dropout_keep_prob
    )
