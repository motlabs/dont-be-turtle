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
from os import getcwd
from os import chdir

import cv2

chdir('..')
sys.path.insert(0,getcwd())
print ('getcwd() = %s' % getcwd())

from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import DATASET_DIR
sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)


import data_loader_tpu

import tensorflow as tf
import numpy as np
from glob import glob



class DataLoaderTest(tf.test.TestCase):


    def test_read_tfrecords_test(self):
        '''
            This test checks below:
            - whether tfrecord is correclty read
        '''

        filenames = tf.placeholder(tf.string, shape=[None])

        dataset_train, dataset_eval = \
            [data_loader_tpu.DataSetInput(
                is_training=is_training,
                data_dir=DATASET_DIR,
                transpose_input=False,
                use_bfloat16=False) for is_training in [True, False]]

        dataset_train = dataset_train.input_fn()
        dataset_eval  = dataset_eval.input_fn()

        iterator_train = dataset_train.make_initializable_iterator()
        iterator_eval  = dataset_eval.make_initializable_iterator()

        # loading tfrecord filenames from self.data_dir
        train_filename_list = glob(DATASET_DIR + '/train-*.*')
        eval_filename_list  = glob(DATASET_DIR + '/eval-*.*')

        print('data_dir = %s' % DATASET_DIR)
        print('train_filename_list = %s'% train_filename_list)
        print('eval_filename_list = %s' % eval_filename_list)



        with self.test_session() as sess:
            sess.run(iterator_train.initializer,
                     feed_dict={filenames: train_filename_list})

            sess.run(iterator_eval.initializer,
                     feed_dict={filenames: eval_filename_list})

            images,labels = iterator_train.get_next()


            img, la = sess.run([images,labels])




if __name__ == '__main__':
    tf.test.main()

