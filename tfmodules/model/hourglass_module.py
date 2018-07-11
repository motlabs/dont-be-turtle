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

from tf_conv_module import get_inception_v2_module
from tf_conv_module import get_separable_conv2d_module
from tf_conv_module import get_linear_bottleneck_module
from tf_conv_module import get_inverted_bottleneck_module
from tf_conv_module import get_residual_module

from tf_deconv_module import get_nearest_neighbor_unpool2d_module
from tf_deconv_module import get_transconv_unpool2d_module



class inception_conv_chout_num(object):

    def __init__(self):

        self.net1 = [96, 96, 32]
        self.net2 = [96, 128]
        self.net3 = [96]



def get_hourglass_conv_module(ch_in,
                             ch_out_num,
                             model_config,
                             stride=1,
                             layer_index=0,
                             scope=None):

    scope       = scope + str(layer_index)
    net         = ch_in
    end_points  = {}
    inception_chout_num_list = inception_conv_chout_num()
    ch_in_num = ch_in.get_shape().as_list()[3]



    with tf.variable_scope(name_or_scope=scope,default_name='hg_conv',values=[ch_in]):

        if model_config.conv_type is 'residual':
            net,end_points = get_residual_module(ch_in         = net,
                                                  ch_out_num    = ch_out_num,
                                                  model_config  = model_config,
                                                  kernel_size   = model_config.kernel_size,
                                                  stride        = stride,
                                                  scope         = model_config.conv_type)

        elif model_config.conv_type is 'inceptionv2':

            net,end_points = get_inception_v2_module(ch_in                     = net,
                                                      inception_conv_chout_num  = inception_chout_num_list,
                                                      model_config              = model_config,
                                                      stride                    = stride,
                                                      scope                     = model_config.conv_type)

        elif model_config.conv_type is 'separable_conv2d':

            net,end_points = get_separable_conv2d_module(ch_in         = net,
                                                          ch_out_num    = ch_out_num,
                                                          model_config  = model_config,
                                                          kernel_size   = model_config.kernel_size,
                                                          stride        = stride,
                                                          scope         = model_config.conv_type)

        elif model_config.conv_type is 'linear_bottleneck':

            net,end_points = get_linear_bottleneck_module(ch_in        = net,
                                                          ch_out_num    = ch_out_num,
                                                          model_config  = model_config,
                                                          kernel_size   = model_config.kernel_size,
                                                          stride        = stride,
                                                          scope         = model_config.conv_type)

        elif model_config.conv_type is 'inverted_bottleneck':

            expand_ch_num = np.floor( ch_in_num *1.5)
            net,end_points = get_inverted_bottleneck_module(ch_in         = net,
                                                             ch_out_num    = ch_out_num,
                                                             expand_ch_num = expand_ch_num,
                                                             model_config  = model_config,
                                                             kernel_size   = model_config.kernel_size,
                                                             stride        = stride,
                                                             scope         = model_config.conv_type)

    return net,end_points





def get_hourglass_deconv_module(ch_in,
                                unpool_rate,
                               model_config=None,
                               layer_index=0,
                               scope=None):

    scope       = scope + str(layer_index)
    net         = ch_in
    end_points  = {}

    with tf.variable_scope(name_or_scope=scope,default_name='hg_deconv',values=[ch_in]):

        if model_config.deconv_type is 'nearest_neighbor_unpool':
            net,end_points= get_nearest_neighbor_unpool2d_module(inputs=net,
                                                                 unpool_rate=unpool_rate,
                                                                 scope = model_config.deconv_type)
        elif model_config.deconv_type is 'conv2dtrans_unpool':
            net,end_points = get_transconv_unpool2d_module(inputs=net,
                                                          unpool_rate = unpool_rate,
                                                          model_config=model_config,
                                                          scope= model_config.deconv_type)

    return net,end_points







def get_conv2d_seq(ch_in,
                  ch_out_num,
                  model_config,
                  scope= None):

    net         = ch_in
    end_points  = {}

    with tf.variable_scope(name_or_scope=scope,default_name='conv2d_seq',values=[ch_in]) as sc:
        scope = 'conv2d_seq'

        endpoint_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            kernel_size         = model_config.kernel_size,
                            weights_initializer = model_config.weights_initializer,
                            weights_regularizer = model_config.weights_regularizer,
                            biases_initializer  = model_config.biases_initializer,
                            trainable           = model_config.is_trainable,
                            normalizer_fn       = model_config.normalizer_fn,
                            padding             = 'SAME',
                            stride              = 1,
                            activation_fn       = None,
                            outputs_collections = endpoint_collection):

            with slim.arg_scope([model_config.normalizer_fn],
                                decay           = model_config.batch_norm_decay,
                                fused           = model_config.batch_norm_fused,
                                is_training     = model_config.is_trainable,
                                activation_fn   = model_config.activation_fn):

                # N sequence of conv2d
                for conv_index in range(model_config.num_of_conv):
                    net = slim.conv2d(inputs        = net,
                                      num_outputs   = ch_out_num,
                                      scope         = scope + '_conv2d_' + str(conv_index))

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            endpoint_collection, clear_collection=True)

    end_points[sc.name + '_out']    = net
    end_points[sc.name + '_in']     = ch_in


    return net, end_points