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
from PIL import Image

# custom packages
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import DATASET_DIR
from path_manager import TFRECORD_TESTSET_DIR
from path_manager import TFRECORD_REALSET_DIR
from path_manager import TFRECORD_TESTIMAGE_DIR
from path_manager import DATASET_BUCKET

sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)

from train_config import TRAININGSET_SIZE
from model_config import DEFAULT_INPUT_CHNUM
from train_config import BATCH_SIZE
from train_config import PreprocessingConfig

import data_loader_tpu
from test_fn_and_util import dataset_parser
from test_fn_and_util import argmax_2d

IMAGE_MAX_VALUE = 255.0
preproc_config = PreprocessingConfig()



class DataLoaderTest(tf.test.TestCase):


    def test_jpeg_imagedata(self):

        '''
            This test check below:
            - given jpeg data set is decoderable
        '''
        image_in = tf.placeholder(dtype=tf.uint8)

        jpeg_tfencode_op = tf.image.encode_jpeg(image=image_in,
                                                    format='rgb',
                                                    quality=100)

        image_byte_in = tf.placeholder(dtype=tf.string)
        jpeg_tfdecode_op = tf.image.decode_jpeg(contents=image_byte_in,
                                              channels=DEFAULT_INPUT_CHNUM)



        # jpegfile_list = glob(DATASET_DIR + '/traintest/lsp/images/*.jp*')
        jpegfile_list = glob(DATASET_DIR + '/testimages/images/*.jp*')
        print('---------------------------------------------------------')
        print ('\n[test_jpeg_imagedata] jpegfile list = %s' % jpegfile_list)

        with self.test_session() as sess:

            for filename in jpegfile_list:
                print ('[test_jpeg_imagedata] current filename = %s' % filename)

                image       = Image.open(filename)
                image_numpy = np.array(image).astype(np.uint8)

                with tf.gfile.FastGFile(filename, 'r') as f:
                    image_data = f.read()

                image_byte  = sess.run([jpeg_tfencode_op],
                                       feed_dict={image_in:image_numpy})
                print ('[test_jpeg_imagedata] tf encoding successful')

                # conversion from raw byte
                image_numpy2= sess.run([jpeg_tfdecode_op],\
                                        feed_dict={image_byte_in:image_data})
                print ('[test_jpeg_imagedata] tf jpeg decoding suceessful')
                print ('[test_jpeg_imagedata] image_numpy2[0] shape = %s'% str(image_numpy2[0].shape))

                # # # conversion from tf encoded jpeg byte ============
                ## tf jpeg decoding from tf encoded byte data does not work (180720)
                # image_numpy3= sess.run([jpeg_tfdecode_op],\
                #                         feed_dict={image_byte_in:image_byte})
                # # #=========================
                # plt.imshow(image_numpy)
                # plt.show()
                # plt.imshow(image_numpy2[0])
                # plt.show()




    def test_read_tfrecords(self):
        '''
            This test checks below:
            - whether tfrecord is correclty read
        '''
        # loading tfrecord filenames from self.data_dir
        train_filename_list = glob(TFRECORD_TESTIMAGE_DIR + '/train-*.*')
        # train_filename_list = glob(TFRECORD_TESTSET_DIR + '/train-*.*')
        # train_filename_list = glob(TFRECORD_REALSET_DIR + '/train-*.*')

        print('\n---------------------------------------------------------')
        print('[test_read_tfrecords] data_dir = %s' % TFRECORD_TESTIMAGE_DIR)
        print('[test_read_tfrecords] train_filename_list = %s'% train_filename_list)


        filenames = tf.placeholder(tf.string)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=TRAININGSET_SIZE)

        # dataset = dataset.map(dataset_parser)
        # dataset = dataset.batch(BATCH_SIZE)
        # dataset = dataset.prefetch(2*BATCH_SIZE)

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(map_func=dataset_parser,
                                          batch_size=BATCH_SIZE,
                                          drop_remainder=True))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        iterator_train = dataset.make_initializable_iterator()

        # jpeg byte tf decoder
        image_byte_input = tf.placeholder(tf.string)
        image_decode_op = tf.image.decode_jpeg(contents=image_byte_input,
                                               channels=DEFAULT_INPUT_CHNUM)

        with self.test_session() as sess:
            sess.run(iterator_train.initializer,
                     feed_dict={filenames: train_filename_list})

            images_byte_op,labels_op,shape_op,stat_op = iterator_train.get_next()

            # here img and la include a batch num of data
            # IMAGE shape = [BATCH_SIZE, HEIGHT, WIDTH ]
            # LABEL SHAPE = [BATCH_SIZE,
            img, la, sha,stat = sess.run([images_byte_op,
                                          labels_op,
                                          shape_op,
                                          stat_op])

            image_numpy = sess.run([image_decode_op],feed_dict={image_byte_input:img[0]})

            # la[batch_index][list_index]
            print ('[test_read_tfrecords] orig image shape = %s' % sha[0])
            print ('[test_read_tfrecords] orig image stat = %s' % stat[0])
            print ('[test_read_tfrecords] label_head_x = %s' % la[0])
            print ('[test_read_tfrecords] label_head_x = %s' % la[0])

            # show first image from the batch data
            # plt.imshow(image_numpy[0])
            # plt.show()




    def test_data_loader_tpu(self):
        '''
            This test checks below:
            - whether tfrecord is correctly read
        '''

        # datadir = TFRECORD_TESTIMAGE_DIR
        datadir = TFRECORD_TESTSET_DIR
        # datadir = DATASET_BUCKET
        print('\n---------------------------------------------------------')
        print('[test_data_loader_tpu] data_dir = %s' % datadir)

        filenames = tf.placeholder(tf.string, shape=[None])
        # dataset_train, dataset_eval = \
        #     [data_loader_tpu.DataSetInput(
        #         is_training=is_training,
        #         data_dir=datadir,
        #         transpose_input=False,
        #         use_bfloat16=False) for is_training in [True, False]]

        dataset_train = \
            data_loader_tpu.DataSetInput(
                is_training=True,
                data_dir=datadir,
                transpose_input=False,
                use_bfloat16=False)

        dataset = dataset_train
        dataset                 = dataset.input_fn()
        iterator_train          = dataset.make_initializable_iterator()
        feature_op, labels_op   = iterator_train.get_next()
        argmax_2d_head_op       = argmax_2d(tensor=labels_op[:, :, :, 0:1])

        favorite_image_index = 5

        with self.test_session() as sess:
            sess.run(iterator_train.initializer)

            for n in range(0,50):

                # argmax2d find coordinate of head
                # containing one heatmap
                feature_numpy, labels_numpy, coord_head_numpy   \
                    = sess.run([feature_op,labels_op,argmax_2d_head_op])

                # some post processing
                image_head          = feature_numpy[favorite_image_index,:,:,:]

                # 256 to 64
                image_head_resized  = cv2.resize(image_head.astype(np.uint8),
                                               dsize=(64,64),
                                               interpolation=cv2.INTER_CUBIC)

                keypoint_head = coord_head_numpy[favorite_image_index].astype(np.uint8)

                # marking the annotation
                # keypoint_head[0] : x
                # keypoint_head[1] : y
                image_head_resized[keypoint_head[1],keypoint_head[0],0] = IMAGE_MAX_VALUE
                image_head_resized[keypoint_head[1],keypoint_head[0],1] = IMAGE_MAX_VALUE
                image_head_resized[keypoint_head[1],keypoint_head[0],2] = IMAGE_MAX_VALUE


                labels_head_numpy       = labels_numpy[favorite_image_index, :, :, 0] \
                                          * IMAGE_MAX_VALUE

                print ('[test_data_loader_tpu] keypoint_head_x0 = %s' % keypoint_head[0])
                print ('[test_data_loader_tpu] keypoint_head_y0 = %s' % keypoint_head[1])

                print('---------------------------------------------------------')

                plt.imshow(feature_numpy[favorite_image_index].astype(np.uint8))
                plt.show()
                plt.imshow(image_head_resized.astype(np.uint8))
                plt.show()
                plt.imshow(labels_head_numpy.astype(np.uint8))
                plt.show()



if __name__ == '__main__':
    tf.test.main()

