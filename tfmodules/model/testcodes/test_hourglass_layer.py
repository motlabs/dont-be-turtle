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

import six
import sys
from datetime import datetime
import tensorflow as tf

from path_manager import TF_MODEL_DIR
from path_manager import TF_LAYER_TEST_DIR

sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,TF_LAYER_TEST_DIR)

from test_layer_util  import create_test_input
from test_layer_util  import get_layer
from test_layer_util  import LayerEndpointName
from test_layer_util  import LayerTestConfig
from test_layer_util  import ModelTestConfig

# where we adopt the NHWC format.

class HourGlassModuleTest(tf.test.TestCase):

    def test_midpoint_name_shape(self):
        '''
            This test checks below:
            - whether name and shape are correctly set.
        '''

        ch_in_num       = 256
        ch_out_num      = 256
        model_config    = ModelConfig()




    # def test_unknown_batchsize_shape(self):
    #     '''
    #         This test check the below case:
    #         - when a module is built without specifying batch_norm size,
    #         check whether the model output has a proper batch_size given by an input
    #     '''


if __name__ == '__main__':
    tf.test.main()

