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
import sys

from path_manager import TF_MODEL_DIR
sys.path.insert(0,TF_MODEL_DIR)

from hourglass_module import get_hourglass_conv_layer
from hourglass_module import get_hourglass_deconv_layer
from hourglass_module import get_conv2d_seq


def get_hourglass_layer(ch_in,
                        model_config,
                        stride                      = 1,
                        conv_kernel_size            = 3,
                        conv_type                   ='residual',
                        deconv_type                 ='nearest_neighbor_unpool',
                        num_of_stacking             = 4,
                        num_of_convseq_atbottom     = 3,
                        pooling_rate                = 2,
                        layer_index                 = 0,
                        scope                       = None):

    scope   = scope + str(layer_index)
    net     = ch_in

    end_points = {}
    with tf.variable_scope(name_or_scope=scope,default_name='hglayer',values=[ch_in]) as sc:

        scope = 'hg_conv'
        end_points[sc.name + '_in'] = net

        # bottem-up convolutional blocks
        # 64(256) --conv+pool-->
        # 16(256) --conv+pool-->
        # 8(256)  --conv+pool-->
        # 4(256)
        with slim.arg_scope([get_hourglass_conv_layer],
                            model_config= model_config,
                            kernel_size = conv_kernel_size,
                            stride      = stride,
                            conv_type   = conv_type):

            # for shorcut connection
            net_array   = tf.TensorArray(dtype= model_config.dtype,
                                         size = num_of_stacking)

            for conv_index in range(0,num_of_stacking):

                net_shape   = net.get_shape().as_list()
                ch_out_num  = net_shape[3]
                net,conv_end_points = get_hourglass_conv_layer(ch_in        = net,
                                                               ch_out_num   = ch_out_num,
                                                               layer_index  = conv_index,
                                                               scope        = scope)

                # max pooling when only stride < 2 where stride is an integer
                if (stride < 2):
                    net = slim.max_pool2d(inputs=net,
                                          kernel_size=[3,3],
                                          stride     = pooling_rate,
                                          padding    = 'VALID',
                                          scope      = scope + '_maxpool' + str(conv_index))
                    conv_end_points[scope + '_maxpool' + str(conv_index)] = net
                else:
                    pooling_rate = stride


                # store tf tensor for shortcut connection
                net_array   = net_array.write(index=conv_index,value=net)

                # end points update
                end_points.update(conv_end_points)

        # A sequence of convolutional block at the bottom
        # 4(256) --conv-->
        # 4(256) --conv-->
        # 4(256) --conv-->
        # 4(256)
        net_shape_at_bottom     = net.get_shape().as_list()
        ch_out_num_at_bottom    = net_shape_at_bottom[3]

        with slim.arg_scope([get_conv2d_seq],
                            ch_out_num      = ch_out_num_at_bottom,
                            num_of_conv     = num_of_convseq_atbottom,
                            model_config    = model_config,
                            kernel_size     = conv_kernel_size):

            scope = 'hg_convseq'
            net,convseq_end_points= get_conv2d_seq(ch_in = net,scope  = scope)
            end_points.update(convseq_end_points)


        # Top- down deconvolutional blocks
        # 4(256) + shortcut --unpool-->
        # 8(256) + shortcut --unpool-->
        # 16(256) + shortcut --unpool-->
        # 64(256)
        with slim.arg_scope([get_hourglass_deconv_layer],
                            unpool_rate     = pooling_rate,
                            deconv_type     = deconv_type,
                            model_config    = model_config):

            for deconv_index in range(0, num_of_stacking):

                # 1) elementwise sum for shortcut connection
                net = tf.add(x=net,
                             y=net_array.read(index=num_of_stacking - deconv_index - 1),
                             name=scope + '_shortcut_sum' + str(deconv_index))
                end_points[scope + '_shortcut_sum' + str(deconv_index)] = net

                # 2) unpooling
                net,deconv_end_points = get_hourglass_deconv_layer(ch_in = net,
                                                                   layer_index=deconv_index,
                                                                   scope=scope)
                # 3) end point update
                end_points.update(deconv_end_points)


        end_points[sc.name + '_out'] = net


    return net, end_points






