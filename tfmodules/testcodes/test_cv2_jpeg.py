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

import sys
from os import getcwd
from os import chdir

chdir('..')
sys.path.insert(0,getcwd())
print ('getcwd() = %s' % getcwd())

import tensorflow as tf
from glob import glob

# image processing tools
import cv2

# custom packages
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import DATASET_DIR


sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)


IMAGE_MAX_VALUE = 255.0


class CV2JpegTest(tf.test.TestCase):


    def test_jpeg_cv2test(self):
        # jpegfile_list = glob(DATASET_DIR + '/traintest/lsp/images/*.jp*')
        jpegfile_list = glob(DATASET_DIR + '/testimages/images/*.jp*')

        print('\n[test_jpeg_cv2test] jpegfile list = %s' % jpegfile_list)


        # opencv test
        for jpegfilename in jpegfile_list:
            image_cv2 = cv2.imread(jpegfilename)
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            cv2.imshow('testimage', image_cv2)
            cv2.waitKey()
