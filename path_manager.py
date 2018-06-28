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
# ! /usr/bin/env python
'''
    filename: path_manager.py
    description: this module include all path information on this proj

    - Author : jaewook Kang @ 20180613

'''

from os import getcwd



PROJ_HOME = getcwd()
TF_MODULE_DIR           = PROJ_HOME              + '/tfmodules'
TF_MODEL_DIR            = TF_MODULE_DIR          + '/model'
TF_LAYER_TEST_DIR       = TF_MODEL_DIR           + '/testcodes'

TF_CNN_MODULE_DIR       = TF_MODEL_DIR           + '/tf-cnn-model'
TF_CNN_TEST_DIR         = TF_CNN_MODULE_DIR      + '/testcodes'
EXPORT_DIR              = PROJ_HOME              + '/exportfiles'
TENSORBOARD_DIR         = EXPORT_DIR             + '/tf_logs'


