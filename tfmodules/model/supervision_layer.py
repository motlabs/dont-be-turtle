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


def get_supervision_layer(ch_in,
                          model_config,
                          layer_index=0,
                          scope=None):
    '''
    ch_in (64x64x256) --> 1x1 conv (64x64x??)

    # heatmap gen path
    --> 1x1 conv (64x64x4)   --> 1x1 conv (64x64x256) --> residual sum (64x64x256)

    # the original path
    --> 1x1 conv (64x64x256) --> residual sum (64x64x256) --> ch_out
    '''

    scope   = scope + str(layer_index)
    net     = ch_in

    with tf.variable_scope(name_or_scope=scope,default_name='svlayer',values=[ch_in]) as sc:

        endpoint_collection = sc.original_name_scope + '_end_points'

        with slim.arg_scope([slim.conv2d],
                            weights_initializer = model_config.weights_initializer,
                            weights_regularizer = model_config.weights_regularizer,
                            biases_initializer = model_config.biases_initializer,
                            trainable          = model_config.is_trainable,
                            kernel_size        = [1, 1],
                            padding            = 'SAME',
                            stride             = 1,
                            normalizer_fn      = model_config.normalizer_fn,
                            activation_fn      = None,
                            outputs_collections=endpoint_collection):

            # batch norm config
            with slim.arg_scope([model_config.normalizer_fn],
                                decay=model_config.batch_norm_decay,
                                fused=model_config.batch_norm_fused,
                                is_training=model_config.is_trainable,
                                activation_fn=model_config.activation_fn):

                # the first 1x1 conv just after hourglass output
                net = slim.conv2d(inputs        = net,
                                  num_outputs   = model_config.num_of_1st1x1conv_ch,
                                  scope         = scope + '_conv1x1_0')

                # intermediate heatmap generation
                heatmaps_out = slim.conv2d(inputs      = net,
                                           num_outputs = model_config.num_of_heatmaps,
                                           scope       = scope + '_conv1x1_heapmatgen_out')

                heatmaps_expansion  = slim.conv2d(inputs        = heatmaps_out,
                                                  num_outputs   = model_config.num_of_channels_out,
                                                  scope         = scope + '_conv1x1_heatmapexp')

                # the original data path
                out = slim.conv2d(inputs        = net,
                                  num_outputs   = model_config.num_of_channels_out,
                                  scope         = scope + '_conv1x1_1')

                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    endpoint_collection, clear_collection=True)

        # combine heatmap expansion to the original data path
        out = tf.add(x = out,
                     y = heatmaps_expansion,
                     name = sc.name + '_out')

        end_points[sc.name + '_out']    = out
        end_points[sc.name + '_in']     = ch_in

    return out, end_points, heatmaps_out



