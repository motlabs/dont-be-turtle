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
from os import chdir

# move to project home directory
chdir('..')

PROJ_HOME               = getcwd()
TF_MODULE_DIR           = PROJ_HOME              + '/tfmodules'

print("[pathmanager] PROJ HOME = %s" % PROJ_HOME)
# tf module related directory
TF_MODEL_DIR            = TF_MODULE_DIR          + '/model'
EXPORT_DIR              = TF_MODULE_DIR          + '/export'

EXPORT_SAVEMODEL_DIR    = EXPORT_DIR             + '/savedmodel'
EXPORT_MODEL_DIR        = EXPORT_DIR             + '/model'
EXPORT_TFLOG_DIR        = EXPORT_DIR             + '/tf_logs'


# sub directory for model
TF_LAYER_TEST_DIR       = TF_MODEL_DIR           + '/testcodes'
TF_CNN_MODULE_DIR       = TF_MODEL_DIR           + '/tf-cnn-model'
TF_CNN_TEST_DIR         = TF_CNN_MODULE_DIR      + '/testcodes'

# data path
DATASET_DIR                 = PROJ_HOME     + '/dataset'
TFRECORD_REALSET_DIR        = DATASET_DIR   + '/tfrecords/realdataset/'
TFRECORD_TESTSET_DIR        = DATASET_DIR   + '/tfrecords/testdataset/'
TFRECORD_TESTIMAGE_DIR      = DATASET_DIR   + '/tfrecords/testimagedataset'
# TRAIN_DATASET_DIR   = DATASET_DIR   + '/train/'
# TEST_DATASET_SIR    = DATASET_DIR   + '/eval/'

# GCP BUCKET ADDRESS
DATASET_BUCKET          = 'gs://pose_dataset_tfrecord/tfrecords/testdataset'
# DATASET_BUCKET          = 'gs://pose_dataset_tfrecord/tfrecords/realdataset'

MODEL_BUCKET             = 'gs://dontbeturtle_ckpt'
TENSORBOARD_BUCKET      = 'gs://dontbeturtle_tflogs'


