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
import numpy as np

def get_output_layer(ch_in,
                     model_config,
                     scope=None):

    net = ch_in

    with tf.variable_scope(name_or_scope=scope,default_name='outlayer',values=[ch_in]) as sc:

        endpoint_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[1, 1],
                            weights_initializer  = model_config.weights_initializer,
                            weights_regularizer  = model_config.weights_regularizer,
                            biases_initializer  = model_config.biases_initializer,
                            normalizer_fn       = model_config.normalizer_fn,
                            activation_fn       = None,
                            outputs_collections = endpoint_collection):
            
            with slim.arg_scope([model_config.normalizer_fn],
                                decay=model_config.batch_norm_decay,
                                fused=model_config.batch_norm_fused,
                                is_training=model_config.is_trainable,
                                activation_fn=model_config.activation_fn,
                                outputs_collections=endpoint_collection):

                for conv_index in range(0,model_config.num_stacking_1x1conv-1):
                    ch_in_num   = net.get_shape().as_list()[3]
                    num_ch_out  = np.floor(ch_in_num * model_config.dim_reduct_ratio)

                    net = slim.conv2d(inputs     = net,
                                      num_outputs= num_ch_out,
                                      scope      = scope + '_conv1x1_' + str(conv_index))

                net = slim.conv2d(inputs        = net,
                                  num_outputs   = model_config.num_of_channels_out,
                                  scope         = scope + '_conv1x1_out')

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            endpoint_collection, clear_collection=True)

        out = tf.identity(input =   net,
                          name  =   sc.name + '_out')
        end_points[sc.name + '_out'] = out
        end_points[sc.name + '_in']  = ch_in

    return out, end_points


