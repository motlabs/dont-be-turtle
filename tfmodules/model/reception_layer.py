# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from hourglass_module import get_hourglass_conv_module


def get_reception_layer(ch_in,
                        model_config,
                        scope=None):

    net = ch_in
    with tf.variable_scope(name_or_scope=scope,default_name='rclayer',values=[ch_in]) as sc:

        endpoint_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,model_config.normalizer_fn],
                            outputs_collections=endpoint_collection):
            # 7x7 conv
            net = slim.conv2d(inputs                = net,
                              num_outputs           = model_config.num_of_channels_out,
                              weights_initializer   = model_config.weights_initializer,
                              weights_regularizer   = model_config.weights_regularizer,
                              biases_initializer    = model_config.biases_initializer,
                              kernel_size           = [model_config.kernel_size,model_config.kernel_size],
                              padding               = 'SAME',
                              stride                = [2,2],
                              normalizer_fn         = None,
                              activation_fn         = None,
                              scope                 = scope + '_conv7x7_out')

            # batch_norm
            net = model_config.normalizer_fn(inputs         = net,
                                             decay          = model_config.batch_norm_decay,
                                             fused          = model_config.batch_norm_fused,
                                             is_training    = model_config.is_trainable,
                                             activation_fn  = model_config.activation_fn,
                                             scope          = scope +'_conv7x7_batchnorm_out')

            # receptive convolutional block
            net, receptconv_end_points = get_hourglass_conv_module(ch_in        = net,
                                                                   ch_out_num   = model_config.num_of_channels_out,
                                                                   stride       = 1,
                                                                   model_config = model_config.conv_config,
                                                                   scope        = scope + '_receptconv')

            # max pooling
            net =  slim.max_pool2d(inputs           =net,
                                   kernel_size      =[3,3],
                                   stride           = [2,2],
                                   padding          ='SAME',
                                   scope            = scope +'_maxpool3x3_out')

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            endpoint_collection, clear_collection=True)


        end_points.update(receptconv_end_points)

        net = tf.identity(input =net,
                          name  = sc.name + '_out')
        end_points[sc.name + '_in']  = ch_in
        end_points[sc.name + '_out'] = net

    return net, end_points