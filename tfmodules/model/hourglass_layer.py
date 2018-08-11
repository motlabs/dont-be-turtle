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
from hourglass_module import get_hourglass_deconv_module
from hourglass_module import get_hourglass_convbottom_module
from tf_conv_module import get_inverted_bottleneck_module
from tf_conv_module import get_linear_bottleneck_module
import numpy as np


def get_hourglass_layer(ch_in,
                        model_config,
                        layer_index= 0,
                        scope= None):

    scope       = scope + str(layer_index)
    net         = ch_in
    shortcut_array   = [] # for shorcut connection
    end_points  = {}
    with tf.variable_scope(name_or_scope=scope,default_name='hglayer',values=[ch_in]) as sc:

        scope = 'hg_conv'
        end_points[sc.name + '_in'] = net

        # set stride by pooling type
        if model_config.pooling_type is 'maxpool':
            stride = 1
        elif model_config.pooling_type is 'convpool':
            stride = model_config.pooling_factor
        else:
            stride = 1

        #----------------------------------------
        # bottem-up convolutional blocks
        for conv_index in range(0,model_config.num_of_stage):

            net_shape   = net.get_shape().as_list()
            ch_out_num  = net_shape[3]
            # print ('[hglayer] conv_index = %s'% conv_index)
            net,conv_end_points = get_hourglass_conv_module(ch_in       = net,
                                                           ch_out_num   = ch_out_num,
                                                           stride       = stride,
                                                           model_config = model_config.conv_config,
                                                           layer_index  = conv_index,
                                                           scope        = scope)

            # max pooling when only stride < 2 where stride is an integer
            if model_config.pooling_type is 'maxpool':
                net = slim.max_pool2d(inputs=net,
                                      kernel_size= [3,3],
                                      stride     = [model_config.pooling_factor,model_config.pooling_factor],
                                      padding    = 'SAME',
                                      scope      = scope + '_maxpool' + str(conv_index))
                conv_end_points[scope + '_maxpool' + str(conv_index)] = net



            # shortcut connections
            if model_config.is_hglayer_shortcut_conv:
                with tf.variable_scope(name_or_scope='shortcut_conv' + str(conv_index)):
                    # shortcut connection with convolution
                    expand_ch_num = np.floor(ch_out_num * model_config.invbottle_expansion_rate)
                    shortcut    = net

                    for shortcut_conv_index in range(0,model_config.num_of_shorcut_invbottleneck_stacking):
                        # stacking of inverted bottleneck blocks
                        shortcut,end_points_shortcut = get_inverted_bottleneck_module(ch_in         =shortcut,
                                                                             ch_out_num     =ch_out_num,
                                                                             expand_ch_num  =expand_ch_num,
                                                                             model_config   =model_config.conv_config,
                                                                             scope=scope + '_shortcut_' + str(conv_index)+str(shortcut_conv_index))
                        end_points.update(end_points_shortcut)

                    # adding linear bottleneck block at the end of the shortcut
                    shortcut,end_points_shortcut = get_linear_bottleneck_module(ch_in   = shortcut,
                                                                        ch_out_num      =ch_out_num,
                                                                        model_config    =model_config.conv_config,
                                                                        scope=scope + '_shortcut_' + str(conv_index) +'linearbottle')
                    end_points.update(end_points_shortcut)
                    shortcut_array.append(shortcut)

            else:
                # just adding
                shortcut_array.append(net)

            # # end points update
            end_points.update(conv_end_points)

        #----------------------------------------
        # A sequence of convolutional block at the bottom
        net_shape_at_bottom     = net.get_shape().as_list()
        ch_out_num_at_bottom    = net_shape_at_bottom[3]

        scope = 'hg_convbottom'
        net,convseq_end_points= get_hourglass_convbottom_module(ch_in        = net,
                                                               ch_out_num   = ch_out_num_at_bottom,
                                                               model_config = model_config.convseq_config,
                                                               scope        = scope)

        end_points.update(convseq_end_points)

        #----------------------------------------
        # Top- down deconvolutional blocks

        scope = 'hg_deconv'
        for deconv_index in range(0, model_config.num_of_stage):
            # print ('[hglayer] deconv_index = %s'% deconv_index)

            # 1) elementwise sum for shortcut connection
            net = tf.add(x=net,
                         y=shortcut_array.pop(),
                         name=scope + '_shortcut_sum' + str(deconv_index))
            end_points[scope + '_shortcut_sum' + str(deconv_index)] = net

            # 2) unpooling
            net,deconv_end_points = get_hourglass_deconv_module(ch_in           = net,
                                                                unpool_rate     = model_config.pooling_factor,
                                                                model_config    = model_config.deconv_config,
                                                                layer_index     = deconv_index,
                                                                is_conv_after_resize = model_config.is_hglayer_conv_after_resize,
                                                                scope           = scope)

            # 3) end point update
            end_points.update(deconv_end_points)

        net = tf.identity(input=net,name=sc.name + '_out' )
        end_points[sc.name + '_out'] = net


    return net, end_points






