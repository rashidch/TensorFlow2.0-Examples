#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf
import core.common as common


def darknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32))
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def darknet_tiny(input_data):
    
    x = common.conv2d(input_layer=input_data, filter_shape=(3,3,3,16))
    x = common.max_pool(x) # 208x208
    x = common.conv2d(input_layer=x, filter_shape=(3,3,16,32))
    x = common.max_pool(x) # 104x104
    x = common.conv2d(input_layer=x, filter_shape=(3,3,32,64))
    x = common.max_pool(x) # 52x52
    x = common.conv2d(input_layer=x, filter_shape=(3,3,64,128))
    x = common.max_pool(x) # 26x26
    route_1 = x # skip connection
    x = common.conv2d(input_layer=x, filter_shape=(3,3,128,256))
    x = common.max_pool(x) #13x13
    x = common.conv2d(input_layer=x, filter_shape=(3,3,256,512))
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='same')(x) #13x13
    route_2 = common.conv2d(input_layer=x, filter_shape=(3,3,512, 1024))

    return route_1, route_2
