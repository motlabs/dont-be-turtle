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
# ======================
#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim



class ConvModuleConfig(object):

    def __init__(self):

        # for convolution modules===================
        # self.conv_type           = 'inceptionv2'
        # self.conv_type           = 'inverted_bottleneck'
        # self.conv_type           = 'linear_bottleneck'
        # self.conv_type           = 'separable_conv2d'
        self.conv_type              = 'residual'
        self.kernel_size            = 3


        self.is_trainable = True
        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay = 0.999
        self.batch_norm_fused = True




class DeconvModuleConfig(object):
    def __init__(self):

        # for deconvolution modules====================
        self.deconv_type                = 'nearest_neighbor_unpool'

        # for unpooling
        self.is_trainable = True
        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True




class ConvSeqModuleConfig(object):

    def __init__(self):

        self.num_of_conv         = 3
        self.kernel_size         = 3
        self.is_trainable        = True


        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn       = slim.batch_norm
        self.activation_fn       = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True



class HourGlassConfig(object):

    def __init__(self):

        # hourglass layer config

        self.num_of_stacking            = 4
        self.input_output_width         = 64
        self.input_output_height        = 64
        self.num_of_channels_out        = 256
        self.is_trainable               = True

        self.conv_config    = ConvModuleConfig()
        self.deconv_config  = DeconvModuleConfig()
        self.convseq_config = ConvSeqModuleConfig()


        self.pooling_type           = 'maxpool'
        # self.pooling_type         = 'convpool'
        self.pooling_factor         = 2




class SupervisionConfig(object):

    def __init__(self):

        self.lossfn_enable          = False
        self.input_output_width     = 64
        self.input_output_height    = 64
        self.num_of_channels_out    = 256

        self.num_of_1st1x1conv_ch   = 256
        self.num_of_heatmaps        = 4
        self.is_trainable           = True


        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True




class ReceptionConfig(object):

    def __init__(self):
        self.input_width     = 256
        self.input_height    = 256

        self.output_width           = 64
        self.output_height          = 64
        self.num_of_channels_out    = 256
        self.is_trainable           = True

        # the kernel_size of the first conv block
        self.kernel_size            = 7


        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True

        self.conv_config    = ConvModuleConfig()




class OutputConfig(object):

    def __init__(self):
        self.input_width            = 64
        self.input_height           = 64
        self.num_of_channels_out    = 4

        self.dim_reduct_ratio              = 1
        self.num_stacking_1x1conv          = 2
        self.is_trainable                  = True

        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True




class ModelConfig(object):

    def __init__(self):
        # common
        self.depth_multiplier   = 1.0
        self.resol_multiplier   = 1.0

        self.dtype              = tf.float32

        self.hg_config          = HourGlassConfig()
        self.sv_config          = SupervisionConfig()
        self.rc_config          = ReceptionConfig()
        self.out_config         = OutputConfig()

