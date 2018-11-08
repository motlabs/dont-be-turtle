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

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# image processing tools
import cv2

# custom packages
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import TFRECORD_TESTSET_DIR
from path_manager import TFRECORD_REALSET_DIR
from path_manager import TFRECORD_TESTIMAGE_DIR

sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)

from model_config import DEFAULT_INPUT_CHNUM
from train_config import TrainConfig
from train_config import PreprocessingConfig

import data_loader_tpu
from preprocessor import preprocess_for_train
from test_fn_and_util import _heatmap_generator
from test_fn_and_util import dataset_parser

IMAGE_MAX_VALUE = 255.0
preproc_config = PreprocessingConfig()
train_config   = TrainConfig()


class PreprocessorTest(tf.test.TestCase):


    def test_preprocessor(self):
        '''
            This test checks below:
            - whether tfrecord is correclty read
        '''
        # loading tfrecord filenames from self.data_dir
        train_filename_list = glob(TFRECORD_TESTIMAGE_DIR + '/train-*.*')
        # train_filename_list = glob(TFRECORD_TESTSET_DIR + '/train-*.*')
        # train_filename_list = glob(TFRECORD_REALSET_DIR + '/train-*.*')

        print('---------------------------------------------------------')
        print('[test_heatmap_gen] data_dir = %s' % TFRECORD_TESTIMAGE_DIR)
        # print('[test_heatmap_gen] data_dir = %s' % TFRECORD_TESTSET_DIR)
        # print('[test_heatmap_gen] data_dir = %s' % TFRECORD_REALSET_DIR)

        print('[test_heatmap_gen] train_filename_list = %s'% train_filename_list)

        # tf.data ====================
        filenames = tf.placeholder(tf.string)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=train_config.trainset_size)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(map_func=dataset_parser,
                                          batch_size=train_config.batch_size,
                                          drop_remainder=True))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        iterator_train = dataset.make_initializable_iterator()
        images_byte_op, labels_op, shape_op, stat_op = iterator_train.get_next()



        # preprocess training set
        image_byte_input = tf.placeholder(tf.string)
        image_prepro_op, is_flip_op, random_ang_rad_op = preprocess_for_train(image_bytes=image_byte_input,
                                                                        use_bfloat16=False,
                                                                        preproc_config=preproc_config)
        # decode original image
        image_op        = tf.image.decode_jpeg(contents=image_byte_input,
                                               channels=DEFAULT_INPUT_CHNUM)

        # test heatmap gen==========
        label_list_input    = tf.placeholder(dtype=tf.int64)
        orig_height_input   = tf.placeholder(dtype=tf.int32)
        orig_width_input    = tf.placeholder(dtype=tf.int32)
        heatmap_gen_op,heatmap_x0_op,heatmap_y0_op, is_flip_op2, random_ang_rad_op2   \
            = _heatmap_generator(label_list=label_list_input,
                                  image_orig_height=orig_height_input,
                                  image_orig_width=orig_width_input,
                                  is_flip=is_flip_op,
                                  random_ang_rad=random_ang_rad_op)

        # tf sessionb ============
        with self.test_session() as sess:
            sess.run(iterator_train.initializer,
                     feed_dict={filenames: train_filename_list})
            for n in range(0,50):

                # load tfrecords
                img, la, sha,stat = sess.run([images_byte_op,
                                             labels_op,
                                             shape_op,
                                             stat_op])
                # generate origianl image
                image_orig_numpy    = sess.run([image_op],feed_dict={image_byte_input:img[0]})


                # preprocessing and heatmap head gen
                '''
                    la[batch_index][list_index]
                    la[0][0] : head
                    la[0][1] : neck
                    la[0][2] : Rshoulder
                    la[0][3] : Lshoulder
                '''
                image_prepro_numpy,heatmap_numpy,heatmap_x0,heatmap_y0,is_flip,random_ang_rad = \
                                                     sess.run([image_prepro_op,
                                                                heatmap_gen_op,
                                                                heatmap_x0_op,
                                                                heatmap_y0_op,
                                                               is_flip_op2,
                                                               random_ang_rad_op2],
                                                                 feed_dict={image_byte_input:img[0],
                                                                            label_list_input:la[0][0], # head only
                                                                            orig_height_input:sha[0][0],
                                                                            orig_width_input: sha[0][1]})



                print ('[test_heatmap_gen] orig image shape = %s' % sha[0])
                print ('[test_heatmap_gen] orig image stat = %s' % stat[0])
                print ('[test_heatmap_gen] label_head_x = %s' % la[0])

                print ('[test_heatmap_gen] heatmap_x0 = %s' % heatmap_x0)
                print ('[test_heatmap_gen] heatmap_y0 = %s' % heatmap_y0)

                print ('[test_heatmap_gen] is_flip = %s' % is_flip)
                print ('[test_heatmap_gen] random_ang_rad = %s' % random_ang_rad)
                print('---------------------------------------------------------')

                # scaling to see by imshow()
                # heatmaps originally have values in [0,1.0]
                heatmap_numpy = heatmap_numpy*IMAGE_MAX_VALUE
                resized_image = cv2.resize(image_prepro_numpy.astype(np.uint8),
                                              dsize=(64,64),
                                              interpolation=cv2.INTER_CUBIC)

                # marking
                # annoation
                resized_image[heatmap_y0.astype(np.uint8),heatmap_x0.astype(np.uint8),0] = IMAGE_MAX_VALUE
                resized_image[heatmap_y0.astype(np.uint8),heatmap_x0.astype(np.uint8),1] = IMAGE_MAX_VALUE
                resized_image[heatmap_y0.astype(np.uint8),heatmap_x0.astype(np.uint8),2] = IMAGE_MAX_VALUE

                # show first image from the batch data
                plt.imshow(image_orig_numpy[0].astype(np.uint8))
                plt.show()
                plt.imshow(resized_image)
                plt.show()
                plt.imshow(heatmap_numpy.astype(np.uint8))
                plt.show()


if __name__ == '__main__':
    tf.test.main()

