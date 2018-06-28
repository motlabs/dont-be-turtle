# Copyright 2018 Jaewook Kang and JoonHo Lee ({jwkang10, junhoning}@gmail.com)
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
#! /usr/bin/env python
'''
    filename: train_config.py
    description: definition of a class containing model training info.

'''
import tensorflow as tf
import path_manager

TRAININGSET_SIZE     = 5000
VALIDATIONSET_SIZE   = 1000
TESTSET_SIZE         = 1000

class TrainConfig(object):
    def __init__(self):

        self.learning_rate      = 0.01
        self.is_learning_rate_decay = True
        self.learning_rate_decay_rate =0.99
        self.opt_type='Adam'

        self.training_epochs    = 100
        self.minibatch_size     = 1000

        # the number of step between evaluation
        self.display_step   = 5
        self.total_batch    = int(TRAININGSET_SIZE / self.minibatch_size)

        # batch norm config
        self.batch_norm_epsilon = 1E-5
        self.batch_norm_decay   = 0.99
        self.FLAGS              = None

        # FC layer config
        self.dropout_keeprate   = 0.8
        self.fc_layer_l2loss_epsilon = 5E-5

        self.tf_data_type       = tf.float32

        # with respect to model exporting
        self.is_graphdef_save_as_pb = True



