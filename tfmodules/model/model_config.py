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
DEFAULT_INPUT_CHNUM     = 3

DEFAULT_RESO_POOL_RATE_IN_RCEPTION = 4.0
DEFAULT_HG_INOUT_RESOL  = DEFAULT_INPUT_RESOL / DEFAULT_RESO_POOL_RATE_IN_RCEPTION

DEFAULT_LABEL_LENGTH    = 3
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


    def show_info(self):

        tf.logging.info('[conv_config] conv_type = %s' % self.conv_type)
        tf.logging.info('[conv_config] kernel_size = %s' % self.kernel_size)
        tf.logging.info('[conv_config] is_trainable = %s' % self.is_trainable)
        # tf.logging.info('[conv_config] weights_regularizer = %s' % str(self.weights_regularizer))
        tf.logging.info('[conv_config] act_fn = %s' % str(self.activation_fn))
        tf.logging.info('[conv_config] batch_norm decay = %s' % self.batch_norm_decay)



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

    def show_info(self):
        tf.logging.info('[deconv_config] conv_type = %s' % self.deconv_type)
        tf.logging.info('[deconv_config] is_trainable = %s' % self.is_trainable)
        # tf.logging.info('[conv_config] weights_regularizer = %s' % str(self.weights_regularizer))
        tf.logging.info('[deconv_config] act_fn = %s' % str(self.activation_fn))
        tf.logging.info('[deconv_config] batch_norm decay = %s' % self.batch_norm_decay)




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
        self.input_height    = int(DEFAULT_INPUT_RESOL * resol_multiplier)
        self.input_width     = int(DEFAULT_INPUT_RESOL * resol_multiplier)

        self.output_width           = int(self.input_width / DEFAULT_RESO_POOL_RATE_IN_RCEPTION)
        self.output_height          = int(self.input_height / DEFAULT_RESO_POOL_RATE_IN_RCEPTION)
        self.num_of_channels_out    = int(DEFAULT_CHANNEL_NUM * depth_multiplier)

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

        self.conv_type = 'inverted_bottleneck'
        self.conv_config    = ConvModuleConfig(conv_type=self.conv_type)



    def show_info(self):

        tf.logging.info('------------------------')
        tf.logging.info('[RecepLayer] is_trainable = %s' % self.is_trainable)
        tf.logging.info('[RecepLayer] weights_regularizer = %s' % str(self.weights_regularizer))
        tf.logging.info('[RecepLayer] act_fn = %s' % str(self.activation_fn))
        tf.logging.info('[RecepLayer] batch_norm decay = %s' % self.batch_norm_decay)
        self.conv_config.show_info()




class HourGlassConfig(object):

    def __init__(self,depth_multiplier, resol_multiplier):

        # hourglass layer config

        self.num_of_stacking            = 4 # shold be less than or equal to 4
        self.input_output_height        = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_output_width         = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.num_of_channels_out        = int(DEFAULT_CHANNEL_NUM * depth_multiplier)
        self.is_trainable               = True


        # self.conv_type           = 'inceptionv2'
        # self.conv_type           = 'inverted_bottleneck'
        # self.conv_type           = 'linear_bottleneck'
        # self.conv_type           = 'separable_conv2d'

        self.conv_type = 'linear_bottleneck'
        self.deconv_type = 'nearest_neighbor_unpool'

        self.conv_config    = ConvModuleConfig(conv_type=self.conv_type)
        self.deconv_config  = DeconvModuleConfig(deconv_type=self.deconv_type)
        self.convseq_config = ConvSeqModuleConfig()

        self.pooling_type           = 'maxpool'
        # self.pooling_type         = 'convpool'
        self.pooling_factor         = 2


    def show_info(self):
        tf.logging.info('------------------------')
        tf.logging.info('[HGLayer] pooling_type = %s' % self.pooling_type)
        self.conv_config.show_info()
        self.deconv_config.show_info()





class SupervisionConfig(object):

    def __init__(self,depth_multiplier, resol_multiplier):

        self.input_output_height    = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_output_width     = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)

        self.num_of_channels_out    = int(DEFAULT_CHANNEL_NUM * depth_multiplier)
        self.num_of_1st1x1conv_ch   = int(DEFAULT_CHANNEL_NUM * depth_multiplier)
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

    def show_info(self):
        tf.logging.info('------------------------')
        tf.logging.info('[SuperLayer] is_trainable = %s' % self.is_trainable)
        tf.logging.info('[SuperLayer] weights_regularizer = %s' % str(self.weights_regularizer))
        tf.logging.info('[SuperLayer] act_fn = %s' % str(self.activation_fn))
        tf.logging.info('[SuperLayer] batch_norm decay = %s' % self.batch_norm_decay)




class OutputConfig(object):

    def __init__(self, resol_multiplier):
        self.input_height           = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
        self.input_width            = int(DEFAULT_HG_INOUT_RESOL * resol_multiplier)
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

    def show_info(self):
        tf.logging.info('------------------------')
        tf.logging.info('[OutputLayer] dim_reduct_ratio = %s' % self.dim_reduct_ratio)
        tf.logging.info('[OutputLayer] num_stacking_1x1conv = %s' % self.num_stacking_1x1conv)
        tf.logging.info('[OutputLayer] is_trainable = %s' % self.is_trainable)
        tf.logging.info('[OutputLayer] weights_regularizer = %s' % str(self.weights_regularizer))
        tf.logging.info('[OutputLayer] act_fn = %s' % str(self.activation_fn))
        tf.logging.info('[OutputLayer] batch_norm decay = %s' % self.batch_norm_decay)




class ModelConfig(object):

    def __init__(self):
        # common
        self.input_height       = int(DEFAULT_INPUT_RESOL)
        self.input_width        = int(DEFAULT_INPUT_RESOL)
        self.input_channel_num  = int(DEFAULT_INPUT_CHNUM)

        self.depth_multiplier   = 1.0 # 1.0 0.75 0.5 0.25
        self.resol_multiplier   = 1.0 # 1.0 0.75 0.5 0.25
        self.num_of_hgstacking  = 2
        self.num_of_labels      = 4

        self.dtype              = tf.float32

        self.hg_config          = HourGlassConfig   (self.depth_multiplier, self.resol_multiplier)
        self.sv_config          = SupervisionConfig (self.depth_multiplier, self.resol_multiplier)
        self.rc_config          = ReceptionConfig   (self.depth_multiplier, self.resol_multiplier)
        self.out_config         = OutputConfig      (self.depth_multiplier)


    def show_info(self):
        tf.logging.info('---------------------------------------')
        tf.logging.info('[model_config] num of hg stacking = %s' % self.num_of_hgstacking)
        tf.logging.info('[model_config] num of labels      = %s' % self.num_of_labels)
        tf.logging.info('[model_config] depth multiplier = %s' % self.depth_multiplier)
        tf.logging.info('[model_config] resol multiplier = %s' % self.resol_multiplier)

        self.rc_config.show_info()
        self.hg_config.show_info()
        self.sv_config.show_info()
        self.out_config.show_info()


