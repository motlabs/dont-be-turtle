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
import numpy as np

DEFAULT_CHANNEL_NUM     = 256.0
DEFAULT_INPUT_RESOL     = 256.0
DEFAULT_HG_INOUT_RESOL  = DEFAULT_INPUT_RESOL / 4.0
NUM_OF_BODY_PART        = 4

class ConvModuleConfig(object):

    def __init__(self,conv_type='residual'):

        # for convolution modules===================
        self.conv_type              = conv_type
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
    def __init__(self,deconv_type='nearest_neighbor_unpool'):

        # for deconvolution modules====================
        self.deconv_type                = deconv_type

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




class ReceptionConfig(object):

    def __init__(self,depth_multiplier, resol_multiplier):
        self.input_height    = np.floor(DEFAULT_INPUT_RESOL * resol_multiplier)
        self.input_width     = np.floor(DEFAULT_INPUT_RESOL * resol_multiplier)

        self.output_width           = np.floor(self.input_width / 4.0)
        self.output_height          = np.floor(self.input_height / 4.0)
        self.num_of_channels_out    = np.floor(DEFAULT_CHANNEL_NUM * depth_multiplier)

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

        self.conv_config    = ConvModuleConfig(conv_type='linear_bottleneck')





class HourGlassConfig(object):

    def __init__(self,depth_multiplier, resol_multiplier):

        # hourglass layer config

        self.num_of_stacking            = 4 # shold be less than or equal to 4
        self.input_output_height        = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_output_width         = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.num_of_channels_out        = np.floor(DEFAULT_CHANNEL_NUM * depth_multiplier)
        self.is_trainable               = True


        # self.conv_type           = 'inceptionv2'
        # self.conv_type           = 'inverted_bottleneck'
        # self.conv_type           = 'linear_bottleneck'
        # self.conv_type           = 'separable_conv2d'
        self.conv_config    = ConvModuleConfig(conv_type='inverted_bottleneck')
        self.deconv_config  = DeconvModuleConfig(deconv_type='nearest_neighbor_unpool')
        self.convseq_config = ConvSeqModuleConfig()

        self.pooling_type           = 'maxpool'
        # self.pooling_type         = 'convpool'
        self.pooling_factor         = 2





class SupervisionConfig(object):

    def __init__(self,depth_multiplier, resol_multiplier):

        self.input_output_height    = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_output_width     = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)

        self.num_of_channels_out    = np.floor(DEFAULT_CHANNEL_NUM * depth_multiplier)
        self.num_of_1st1x1conv_ch   = np.floor(DEFAULT_CHANNEL_NUM * depth_multiplier)
        self.num_of_heatmaps        = 4

        self.is_trainable           = True
        self.lossfn_enable          = False


        self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.normalizer_fn          = slim.batch_norm
        self.activation_fn          = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True







class OutputConfig(object):

    def __init__(self, resol_multiplier):
        self.input_height           = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_width            = np.floor(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.num_of_channels_out    = NUM_OF_BODY_PART

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
        self.depth_multiplier   = 1.0 # 1.0 0.75 0.5 0.25
        self.resol_multiplier   = 1.0 # 1.0 0.75 0.5 0.25
        self.num_of_hgstacking  = 2
        self.num_of_labels      = 4

        self.dtype              = tf.float32

        self.hg_config          = HourGlassConfig   (self.depth_multiplier, self.resol_multiplier)
        self.sv_config          = SupervisionConfig (self.depth_multiplier, self.resol_multiplier)
        self.rc_config          = ReceptionConfig   (self.depth_multiplier, self.resol_multiplier)
        self.out_config         = OutputConfig      (self.depth_multiplier)

