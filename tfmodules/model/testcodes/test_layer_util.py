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
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    end_points  = {}

    with tf.variable_scope(name_or_scope=scope, default_name='test_layer',values=[ch_in]):

        if layer_type == 'hourglass':
            net, end_points = get_hourglass_layer(ch_in                 =net,
                                                model_config            =model_config,
                                                pooling_factor          =layer_config.pooling_factor,
                                                conv_kernel_size        =layer_config.conv_kernel_size,
                                                pooling_type            =layer_config.pooling_type,
                                                conv_type               =layer_config.conv_type,
                                                deconv_type             =layer_config.deconv_type,
                                                num_of_stacking         =layer_config.num_of_stacking,
                                                num_of_convseq_atbottom =layer_config.num_of_convseq_atbottom,
                                                layer_index             =layer_index,
                                                scope=layer_type)


    return net, end_points


class LayerEndpointName(object):

    # for unittest
    def __init__(self,layer_type,input_shape,output_shape):

        if layer_type is 'hourglass':

            '''
                unittest LayerTestConfig setting
                self.conv_type           = 'residual'
                self.deconv_type         = 'nearest_neighbor_unpool'
                self.pooling_type        = 'maxpool'
    
                self.conv_kernel_size           = 3
                self.pooling_factor             = 2
    
                self.num_of_stacking            = 4
                self.num_of_convseq_atbottom    = 3
    
                self.input_output_width         = 64
                self.input_output_height        = 64
            '''

            self.name_list = [
                    'unittest0/hourglass0_in',
                    'unittest0/hourglass0/hg_conv0/residual_in',
                    'unittest0/hourglass0/hg_conv0/residual_out',
                    'hg_conv_maxpool0',
                    'unittest0/hourglass0/hg_conv1/residual_in',
                    'unittest0/hourglass0/hg_conv1/residual_out',
                    'hg_conv_maxpool1',
                    'unittest0/hourglass0/hg_conv2/residual_in',
                    'unittest0/hourglass0/hg_conv2/residual_out',
                    'hg_conv_maxpool2',
                    'unittest0/hourglass0/hg_conv3/residual_in',
                    'unittest0/hourglass0/hg_conv3/residual_out',
                    'hg_conv_maxpool3',
                    'unittest0/hourglass0/hg_convseq0_in',
                    'unittest0/hourglass0/hg_convseq0_out',
                    'hg_deconv_shortcut_sum0',
                    'unittest0/hourglass0/hg_deconv0/nearest_neighbor_unpool_in',
                    'unittest0/hourglass0/hg_deconv0/nearest_neighbor_unpool_out',
                    'hg_deconv_shortcut_sum1',
                    'unittest0/hourglass0/hg_deconv1/nearest_neighbor_unpool_in',
                    'unittest0/hourglass0/hg_deconv1/nearest_neighbor_unpool_out',
                    'hg_deconv_shortcut_sum2',
                    'unittest0/hourglass0/hg_deconv2/nearest_neighbor_unpool_in',
                    'unittest0/hourglass0/hg_deconv2/nearest_neighbor_unpool_out',
                    'hg_deconv_shortcut_sum3',
                    'unittest0/hourglass0/hg_deconv3/nearest_neighbor_unpool_in',
                    'unittest0/hourglass0/hg_deconv3/nearest_neighbor_unpool_out',
                    'unittest0/hourglass0_out']

            input_shape_hg_conv0    = input_shape
            input_shape_hg_conv1    = [input_shape[0],input_shape[1]/2, input_shape[2]/2,input_shape[3]]
            input_shape_hg_conv2    = [input_shape[0],input_shape[1]/4, input_shape[2]/4,input_shape[3]]
            input_shape_hg_conv3    = [input_shape[0],input_shape[1]/8, input_shape[2]/8,input_shape[3]]
            input_shape_hg_convseq  = [input_shape[0],input_shape[1]/16,input_shape[2]/16,input_shape[3]]
            input_shape_hg_deconv0  = [input_shape[0],input_shape[1]/8, input_shape[2]/8,input_shape[3]]
            input_shape_hg_deconv1  = [input_shape[0],input_shape[1]/4, input_shape[2]/4,input_shape[3]]
            input_shape_hg_deconv2  = [input_shape[0],input_shape[1]/2, input_shape[2]/2,input_shape[3]]
            input_shape_hg_deconv3  = output_shape

            self.shape_dict = {\
                                self.name_list[0]:input_shape,
                                self.name_list[1]:input_shape_hg_conv0,
                                self.name_list[2]:input_shape_hg_conv0,
                                self.name_list[3]:input_shape_hg_conv1,
                                self.name_list[4]:input_shape_hg_conv1,
                                self.name_list[5]:input_shape_hg_conv1,
                                self.name_list[6]:input_shape_hg_conv2,
                                self.name_list[7]:input_shape_hg_conv2,
                                self.name_list[8]:input_shape_hg_conv2,
                                self.name_list[9]:input_shape_hg_conv3,
                                self.name_list[10]:input_shape_hg_conv3,
                                self.name_list[11]:input_shape_hg_conv3,
                                self.name_list[12]:input_shape_hg_convseq,
                                self.name_list[13]:input_shape_hg_convseq,
                                self.name_list[14]:input_shape_hg_convseq,
                                self.name_list[15]:input_shape_hg_convseq,
                                self.name_list[16]:input_shape_hg_convseq,
                                self.name_list[17]:input_shape_hg_deconv0,
                                self.name_list[18]:input_shape_hg_deconv0,
                                self.name_list[19]:input_shape_hg_deconv0,
                                self.name_list[20]:input_shape_hg_deconv1,
                                self.name_list[21]:input_shape_hg_deconv1,
                                self.name_list[22]:input_shape_hg_deconv1,
                                self.name_list[23]:input_shape_hg_deconv2,
                                self.name_list[24]: input_shape_hg_deconv2,
                                self.name_list[25]: input_shape_hg_deconv2,
                                self.name_list[26]: input_shape_hg_deconv3,
                                self.name_list[27]: output_shape}

        # elif layer_type is 'reception':
        #
        # elif layer_type is 'supervision':
        #
        # elif layer_tyupe is 'output':
        #




class LayerTestConfig(object):

    def __init__(self):

        # hourglass layer config
        # self.conv_type           = 'inceptionv2'
        self.conv_type           = 'inverted_bottleneck'
        # self.conv_type           = 'linear_bottleneck'
        # self.conv_type           = 'separable_conv2d'
        # self.conv_type           = 'residual'

        self.deconv_type         = 'nearest_neighbor_unpool'
        self.pooling_type        = 'maxpool'
        # self.pooling_type        = 'convpool'

        self.conv_kernel_size    = 3
        self.pooling_factor        = 2

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