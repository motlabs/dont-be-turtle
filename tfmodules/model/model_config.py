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


class ModelConfig(object):

    def __init__(self):

        # common
        self.depth_multiplier   = 1.0
        self.resol_multiplier   = 1.0

        self.is_trainable       = True
        self.dtype              = tf.float32

        # 1) for convolution layers
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn      = slim.batch_norm
        self.activation_fn      = tf.nn.relu6

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True



        # 2) for deconvolutional layer
        self.unpool_weights_initializer = tf.contrib.layers.xavier_initializer()
        self.unpool_weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.unpool_biases_initializer  = slim.init_ops.zeros_initializer()
        self.unpool_normalizer_fn      = slim.batch_norm
        self.unpool_activation_fn      = tf.nn.relu6

        # batch_norm
        self.unpool_batch_norm_decay   = 0.999
        self.unpool_batch_norm_fused   = True