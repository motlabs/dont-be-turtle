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


import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

from path_manager import TF_MODEL_DIR
from path_manager import TF_LAYER_TEST_DIR

sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,TF_LAYER_TEST_DIR)

from hourglass_layer import get_hourglass_layer
# from reception_layer import get_reception_layer
# from spervision_layer import get_supervision_layer
# from output_layer import get_output_layer


# where we adopt the NHWC format.


def create_test_input(batchsize,heightsize,widthsize,channelnum):

    '''Create test input tensor by tf.placeholder
        input : the size of 4d tensor
        return:
    '''

    if None in [batchsize, heightsize,widthsize,channelnum]:
        return tf.placeholder(tf.float32, [batchsize,heightsize,widthsize,channelnum])
    else:
        return tf.to_float(
            np.tile(
                np.reshape(
                    np.reshape(np.arange(heightsize),[heightsize,1]) +
                    np.reshape(np.arange(widthsize), [1,widthsize]),
                    [1, heightsize,widthsize,1]),
                    [batchsize,1,1,channelnum]))



def get_layer(ch_in,
              layer_config,
              model_config,
              layer_index=0,
              layer_type = 'hourglass',
              scope=None):

    scope       = scope + str(layer_index)
    ch_in_num   = ch_in.get_shape().as_list()[3]
    net         = ch_in

    with tf.variable_scope(name_or_scope=scope, default_name='test_layer',values=[ch_in]):

        if layer_type == 'hourglass':
            net, end_points = get_hourglass_layer(ch_in                 =net,
                                                model_config            =model_config,
                                                stride                  =layer_config.stride,
                                                conv_kernel_size        =layer_config.conv_kernel_size,
                                                conv_type               =layer_config.conv_type,
                                                deconv_type             =layer_config.deconv_type,
                                                num_of_stacking         =layer_config.num_of_stacking,
                                                num_of_convseq_atbottom =layer_config.num_of_convseq_atbottom,
                                                pooling_rate            =layer_config.pooling_rate,
                                                layer_index             =layer_index,
                                                scope=layer_type)


    return net, end_points


class LayerEndpointName(object):

    def __init__(self,input_shape,output_shape):

        self.name_list = ['unittest0/inceptionv2/inceptionv2_net1_conv1x1',
                         'unittest0/inceptionv2/inceptionv2_net1_conv3x3_1',
                         'unittest0/inceptionv2/inceptionv2_net1_conv3x3_2',
                         'unittest0/inceptionv2/inceptionv2_net2_conv1x1',
                         'unittest0/inceptionv2/inceptionv2_net2_conv3x3',
                         'unittest0/inceptionv2/inceptionv2_net3_maxpool3x3',
                         'unittest0/inceptionv2/inceptionv2_net3_conv1x1',
                         'inceptionv2/inceptionv2_concat',
                         'unittest0/inceptionv2_out']

        self.shape_dict = {
                            self.name_list[0]:[input_shape[0],  input_shape[1], input_shape[2],     chnum_list.net1[0]],
                            self.name_list[1]:[input_shape[0],  input_shape[1], input_shape[2],     chnum_list.net1[1]],
                            self.name_list[2]:[output_shape[0], output_shape[1],output_shape[2],    chnum_list.net1[2]],
                            self.name_list[3]:[input_shape[0],  input_shape[1], input_shape[2],     chnum_list.net2[0]],
                            self.name_list[4]:[output_shape[0], output_shape[1],output_shape[2],    chnum_list.net2[1]],
                            self.name_list[5]:[output_shape[0], output_shape[1],output_shape[2],    input_shape[3]],
                            self.name_list[6]:[output_shape[0], output_shape[1],output_shape[2],    chnum_list.net3[0]],
                            self.name_list[7]:output_shape,
                            self.name_list[8]:output_shape,
                            }




class LayerTestConfig(object):

    def __init__(self):

        # hourglass layer config
        self.conv_type           = 'residual'
        self.deconv_type         = 'nearest_neighbor_unpool'

        self.conv_kernel_size    = 3
        self.pooling_rate        = 2

        self.stride                     = 1
        self.num_of_stacking            = 4
        self.num_of_convseq_atbottom    = 3

        self.input_output_width         = 64
        self.input_output_height        = 64





class ModelTestConfig(object):

    def __init__(self):


        # common
        self.depth_multiplier   = 1.0
        self.resol_multiplier   = 1.0

        self.is_trainable       = True
        self.dtype              = tf.float32

        # for convolution layers
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn      = slim.batch_norm
        self.activation_fn      = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True


        # for deconvolution layers
        self.unpool_weights_initializer = tf.contrib.layers.xavier_initializer()
        self.unpool_weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.unpool_biases_initializer  = slim.init_ops.zeros_initializer()
        self.unpool_normalizer_fn      = slim.batch_norm
        self.unpool_activation_fn      = tf.nn.relu6

        # batch_norm
        self.unpool_batch_norm_decay   = 0.999
        self.unpool_batch_norm_fused   = True