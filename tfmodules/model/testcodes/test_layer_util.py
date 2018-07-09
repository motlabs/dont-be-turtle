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

from hourglass_layer    import get_hourglass_layer
from reception_layer    import get_reception_layer
from supervision_layer  import get_supervision_layer
from output_layer       import get_output_layer


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
              model_config,
              layer_index=0,
              layer_type = 'hourglass',
              scope=None):

    scope       = scope + str(layer_index)
    ch_in_num   = ch_in.get_shape().as_list()[3]
    net         = ch_in
    end_points  = {}
    heatmaps_out = None

    with tf.variable_scope(name_or_scope=scope, default_name='test_layer',values=[ch_in]):

        if layer_type == 'hourglass':
            net, end_points = get_hourglass_layer(ch_in                 =net,
                                                  model_config          =model_config,
                                                  layer_index           =layer_index,
                                                  scope=layer_type)

        elif layer_type is 'reception':

            net, end_points = get_reception_layer(ch_in         = net,
                                                  model_config  = model_config,
                                                  scope         = layer_type)

        elif layer_type is 'supervision':

            net, end_points, heatmaps_out = get_supervision_layer(ch_in                 =net,
                                                                  model_config          =model_config,
                                                                  layer_index           =layer_index,
                                                                  scope                 =layer_type)

        elif layer_type is 'output':

            net, end_points = get_output_layer(ch_in            = net,
                                               model_config     = model_config,
                                               scope            = layer_type)


    return net, end_points,heatmaps_out




class LayerEndpointName(object):

    # for unittest
    def __init__(self,layer_type,
                 input_shape,
                 output_shape,
                 conv_type='residual',
                 deconv_type='nearest_neighbor_unpool'):

        if layer_type is 'hourglass':

            '''
                unittest LayerTestConfig configuration
    
                self.pooling_factor             = 2
                self.num_of_stacking            = 4
                self.num_of_convseq_atbottom    = 3
    
            '''

            self.name_list = [
                    'unittest0/hourglass0_in',
                    'unittest0/hourglass0/hg_conv0/'+ conv_type +'_in',
                    'unittest0/hourglass0/hg_conv0/'+ conv_type +'_out',
                    'hg_conv_maxpool0',
                    'unittest0/hourglass0/hg_conv1/'+ conv_type +'_in',
                    'unittest0/hourglass0/hg_conv1/'+ conv_type +'_out',
                    'hg_conv_maxpool1',
                    'unittest0/hourglass0/hg_conv2/'+ conv_type +'_in',
                    'unittest0/hourglass0/hg_conv2/'+ conv_type +'_out',
                    'hg_conv_maxpool2',
                    'unittest0/hourglass0/hg_conv3/'+ conv_type +'_in',
                    'unittest0/hourglass0/hg_conv3/'+ conv_type +'_out',
                    'hg_conv_maxpool3',
                    'unittest0/hourglass0/hg_convseq_in',
                    'unittest0/hourglass0/hg_convseq_out',
                    'hg_deconv_shortcut_sum0',
                    'unittest0/hourglass0/hg_deconv0/'+ deconv_type +'_in',
                    'unittest0/hourglass0/hg_deconv0/'+ deconv_type +'_out',
                    'hg_deconv_shortcut_sum1',
                    'unittest0/hourglass0/hg_deconv1/'+ deconv_type +'_in',
                    'unittest0/hourglass0/hg_deconv1/'+ deconv_type +'_out',
                    'hg_deconv_shortcut_sum2',
                    'unittest0/hourglass0/hg_deconv2/'+ deconv_type +'_in',
                    'unittest0/hourglass0/hg_deconv2/'+ deconv_type +'_out',
                    'hg_deconv_shortcut_sum3',
                    'unittest0/hourglass0/hg_deconv3/'+ deconv_type +'_in',
                    'unittest0/hourglass0/hg_deconv3/'+ deconv_type +'_out',
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

        elif layer_type is 'reception':
            self.name_list  = ['unittest0/reception_in',
                               'unittest0/reception/reception_conv7x7_out',
                               'unittest0/reception/reception_conv7x7_batchnorm_out',
                               'unittest0/reception/reception_maxpool3x3_out',
                               'unittest0/reception_out']

            input_shape_receptconv = [input_shape[0],input_shape[1]/2, input_shape[2]/2,output_shape[3]]
            input_shape_maxpool    = input_shape_receptconv

            self.shape_dict = {self.name_list[0]:input_shape,
                               self.name_list[1]:input_shape_receptconv,
                               self.name_list[2]:input_shape_maxpool,
                               self.name_list[3]:output_shape,
                               self.name_list[4]:output_shape}


        elif layer_type is 'supervision':
            self.name_list = ['unittest0/supervision0_in',
                              'unittest0/supervision0/supervision0_conv1x1_0',
                              'unittest0/supervision0/supervision0_conv1x1_1',
                              'unittest0/supervision0/supervision0_conv1x1_heapmatgen_out',
                              'unittest0/supervision0/supervision0_conv1x1_heatmapexp',
                              'unittest0/supervision0_out']

            self.shape_dict = {self.name_list[0]:input_shape,
                               self.name_list[1]:input_shape,
                               self.name_list[2]:input_shape,
                               self.name_list[3]:[input_shape[0],input_shape[1],input_shape[2],4],
                               self.name_list[4]:input_shape,
                               self.name_list[5]:output_shape}
        elif layer_type is 'output':

            self.name_list = [  'unittest0/output_in',
                                'unittest0/output/output_conv1x1_0',
                                'unittest0/output/output_conv1x1_0/BatchNorm',
                                'unittest0/output/output_conv1x1_out',
                                'unittest0/output/output_conv1x1_out/BatchNorm',
                                'unittest0/output_out']

            self.shape_dict = {
                                self.name_list[0]:input_shape,
                                self.name_list[1]:input_shape,
                                self.name_list[2]:input_shape,
                                self.name_list[3]:output_shape,
                                self.name_list[4]:output_shape,
                                self.name_list[5]:output_shape
                                }


class ModelEndpointName(object):

    # for unittest
    def __init__(self,
                 input_shape,
                 output_shape,
                 hg_ch_num):


        self.name_list = ['model_in',
                         'model/reception/reception_in',
                         'model/reception/reception_out',
                         'model/stacked_hg/hourglass0_in',
                         'model/stacked_hg/hourglass0_out',
                         'model/stacked_hg/supervision0_in',
                         'model/stacked_hg/supervision0/supervision0_conv1x1_heapmatgen_out',
                         'model/stacked_hg/supervision0_out',
                         'model/stacked_hg/hourglass1_in',
                         'model/stacked_hg/hourglass1_out',
                         'model/stacked_hg/supervision1_in',
                         'model/stacked_hg/supervision1/supervision1_conv1x1_heapmatgen_out',
                         'model/stacked_hg/supervision1_out',
                         'model/output/output_in',
                         'model/output/output_out',
                         'model_out']

        hg_shape = [input_shape[0],input_shape[1]/4,input_shape[2]/4,hg_ch_num]

        self.shape_dict = {
            self.name_list[0]: input_shape,
            self.name_list[1]: input_shape,
            self.name_list[2]: hg_shape,
            self.name_list[3]: hg_shape,
            self.name_list[4]: hg_shape,
            self.name_list[5]: hg_shape,
            self.name_list[6]: output_shape,
            self.name_list[7]: hg_shape,
            self.name_list[8]: hg_shape,
            self.name_list[9]: hg_shape,
            self.name_list[10]: hg_shape,
            self.name_list[11]: output_shape,
            self.name_list[12]: hg_shape,
            self.name_list[13]: hg_shape,
            self.name_list[14]: output_shape,
            self.name_list[15]: output_shape
        }




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
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer = slim.init_ops.zeros_initializer()
        self.normalizer_fn = slim.batch_norm
        self.activation_fn = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay = 0.999
        self.batch_norm_fused = True






class DeconvModuleConfig(object):
    def __init__(self):

        # for deconvolution modules====================
        self.deconv_type                = 'nearest_neighbor_unpool'

        # for unpooling
        self.is_trainable = True
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn      = slim.batch_norm
        self.activation_fn      = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True






class ConvSeqModuleConfig(object):

    def __init__(self):

        self.num_of_conv         = 3
        self.is_trainable        = True
        self.kernel_size            = 3


        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn      = slim.batch_norm
        self.activation_fn      = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True






class HourGlassTestConfig(object):

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
        # self.pooling_type        = 'convpool'
        self.pooling_factor         = 2





class SupervisionTestConfig(object):

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





class ReceptionTestConfig(object):

    def __init__(self):
        self.input_width     = 256
        self.input_height    = 256

        self.output_width     = 64
        self.output_height    = 64
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





class OutputTestConfig(object):

    def __init__(self):
        self.input_width     = 64
        self.input_height    = 64
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




class ModelTestConfig(object):

    def __init__(self):
        # common
        self.depth_multiplier   = 1.0
        self.resol_multiplier   = 1.0

        self.dtype              = tf.float32

        self.hg_config          = HourGlassTestConfig()
        self.sv_config          = SupervisionTestConfig()
        self.rc_config          = ReceptionTestConfig()
        self.out_config         = OutputTestConfig()

