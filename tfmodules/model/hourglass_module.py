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
from tf_conv_module import get_conv2d_seq

from tf_deconv_module import get_nearest_neighbor_unpool2d_module
from tf_deconv_module import get_transconv_unpool2d_module
from tf_deconv_module import get_nearest_neighbor_resize_module
from tf_deconv_module import get_bilinear_resize_module
from tf_deconv_module import get_bicubic_resize_module


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

            expand_ch_num = np.floor( ch_in_num * model_config.invbottle_expansion_rate)
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
                               is_conv_after_resize=True,
                               scope=None):

    scope       = scope + str(layer_index)
    net         = ch_in
    end_points  = {}

    with tf.variable_scope(name_or_scope=scope,default_name='hg_deconv',values=[ch_in]):

        '''
            note that only bilinear resize module support tflite conversion (2018 July)
        '''
        if model_config.deconv_type is 'nearest_neighbor_resize':
            net,end_points = get_nearest_neighbor_resize_module(inputs=net,
                                                               resize_rate=unpool_rate,
                                                               scope = model_config.deconv_type)
        elif model_config.deconv_type is 'bilinear_resize':
            net, end_points = get_bilinear_resize_module(inputs=net,
                                                         resize_rate=unpool_rate,
                                                         model_config=model_config,
                                                         is_conv_after_resize=is_conv_after_resize,
                                                         scope= model_config.deconv_type)

        elif model_config.deconv_type is 'bicubic_resize':
            net, end_points = get_bicubic_resize_module(inputs = net,
                                                      resize_rate= unpool_rate,
                                                      scope= model_config.deconv_type)

        elif model_config.deconv_type is 'conv2dtrans_unpool':
            net,end_points = get_transconv_unpool2d_module(inputs=net,
                                                          unpool_rate = unpool_rate,
                                                          model_config=model_config,
                                                          scope= model_config.deconv_type)

        elif model_config.deconv_type is 'nearest_neighbor_unpool':
            net, end_points = get_nearest_neighbor_unpool2d_module(inputs=net,
                                                                   unpool_rate=unpool_rate,
                                                                   scope=model_config.deconv_type)

    return net,end_points






def get_hourglass_convbottom_module(ch_in,
                                    ch_out_num,
                                   model_config=None,
                                   scope=None):
    net         = ch_in
    end_points  = {}

    with tf.variable_scope(name_or_scope=scope,default_name='hg_convbottom',values=[ch_in]) as sc:

        if model_config.conv_type is 'inverted_bottleneck':
            expand_ch_num = np.floor( ch_out_num * model_config.invbottle_expansion_rate)
            net, end_points = get_inverted_bottleneck_module(ch_in          = ch_in,
                                                             ch_out_num     = ch_out_num,
                                                             expand_ch_num  = expand_ch_num,
                                                             model_config   = model_config,
                                                             scope          = model_config.conv_type)

        elif model_config.conv_type is 'conv2d_seq':
            net,end_points = get_conv2d_seq(ch_in           = ch_in,
                                            ch_out_num      = ch_out_num,
                                            model_config    = model_config,
                                            scope           = model_config.conv_type)

        end_points[sc.name + '_out'] = net
        end_points[sc.name + '_in'] = ch_in

    return net,end_points





