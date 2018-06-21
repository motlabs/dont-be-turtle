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

from tfslim_module import get_inception_v2_module
from tfslim_module import get_separable_conv2d_module
from tfslim_module import get_linear_bottleneck_module
from tfslim_module import get_inverted_bottleneck_module
from tfslim_module import get_residual_module

class inception_conv_chout_num(object):

    def __init__(self):

        self.net1 = [96, 96, 32]
        self.net2 = [96, 128]
        self.net3 = [96]


def get_hourglass_conv_layer(ch_in,
                             ch_out_num,
                             model_config,
                             layer_index=0,
                             kernel_size=3,
                             stride=1,
                             conv_type='residual',
                             scope=None):

    scope = scope + '_hg_conv' + str(layer_index)
    net = ch_in

    with tf.name_scope(name=scope,default_name='hg_conv',values=[ch_in]):

        if conv_type == 'residual':
            net = get_residual_module(ch_in         = net,
                                      ch_out_num    = ch_out_num,
                                      model_config  = model_config,
                                      kernel_size   = kernel_size,
                                      stride        = stride,
                                      scope         = scope)

        elif conv_type == 'inceptionv2':

            net = get_inception_v2_module(ch_in                     = net,
                                          inception_conv_chout_num  = inception_conv_chout_num,
                                          model_config              = model_config,
                                          stride                    = stride,
                                          scope                     = scope)

        elif conv_type == 'separable_conv2d':

            net = get_separable_conv2d_module(ch_in         = net,
                                              ch_out_num    = ch_out_num,
                                              model_config  = model_config,
                                              kernel_size   = kernel_size,
                                              stride        = stride,
                                              scope         = scope)
        elif conv_type == 'linear_bottleneck':

            net = get_linear_bottleneck_module(ch_in        = net,
                                              ch_out_num    = ch_out_num,
                                              model_config  = model_config,
                                              kernel_size   = kernel_size,
                                              stride        =stride,
                                              scope         = scope)
        elif conv_type == 'inverted_bottleneck':

            net = get_inverted_bottleneck_module(ch_in         = net,
                                                 ch_out_num    = ch_out_num,
                                                 expand_ch_num = tf.floor(ch_in*1.2),
                                                 model_config  = model_config,
                                                 kernel_size   = kernel_size,
                                                 stride        = stride,
                                                 scope         = scope)

    return net

