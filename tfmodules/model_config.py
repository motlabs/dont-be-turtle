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

import tensorflow as tf
import tensorflow.contrib.slim as slim


class ModelConfig(object):

    def __init__(self):

        self.is_trainable       = True
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)
        self.biases_initializer  = slim.init_ops.zeros_initializer()
        self.normalizer_fn      = slim.batch_norm

        self.activation_fn      = tf.nn.relu6
        self.dtype              = tf.float32

        # batch_norm
        self.batch_norm_decay   = 0.999
        self.batch_norm_fused   = True


        # meta parameter
        self.depth_multiplier   = 1.0
        self.resol_multiplier   = 1.0

