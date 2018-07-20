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
import tensorflow as tf
from PIL import Image

import numpy as np
from glob import glob
import matplotlib.pyplot as plt


chdir('..')
sys.path.insert(0,getcwd())
print ('getcwd() = %s' % getcwd())

from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import DATASET_DIR
from path_manager import TFRECORD_DIR
from path_manager import TFRECORD_TEST_DIR

sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)

from train_config import TRAININGSET_SIZE
from train_config import BATCH_SIZE
from model_config import DEFAULT_INPUT_CHNUM

import data_loader_tpu

import cv2
from PIL import Image


class DataLoaderTest(tf.test.TestCase):


    # def test_jpeg_cv2test(self):
    #     # jpegfile_list = glob(DATASET_DIR + '/traintest/lsp/images/*.jp*')
    #     jpegfile_list = glob(DATASET_DIR + '/testimages/*.jp*')
    #
    #     print('\n[test_jpeg_cv2test] jpegfile list = %s' % jpegfile_list)
    #
    #
    #     # opencv test
    #     for jpegfilename in jpegfile_list:
    #         image_cv2 = cv2.imread(jpegfilename)
    #         cv2.startWindowThread()
    #         cv2.namedWindow("preview")
    #         cv2.imshow('testimage', image_cv2)
    #         cv2.waitKey()


    # def test_jpeg_imagedata(self):
    #
    #     '''
    #         This test check below:
    #         - given jpeg data set is decoderable
    #     '''
    #     # image_in = tf.placeholder(dtype=tf.uint8,shape=[640,480,3])
    #     image_in = tf.placeholder(dtype=tf.uint8)
    #
    #     jpeg_tfencode_op = tf.image.encode_jpeg(image=image_in,
    #                                                 format='rgb',
    #                                                 quality=100)
    #
    #     image_byte_in = tf.placeholder(dtype=tf.string)
    #     jpeg_tfdecode_op = tf.image.decode_jpeg(contents=image_byte_in,
    #                                           channels=DEFAULT_INPUT_CHNUM)
    #
    #
    #
    #     # jpegfile_list = glob(DATASET_DIR + '/traintest/lsp/images/*.jp*')
    #     jpegfile_list = glob(DATASET_DIR + '/testimages/*.jp*')
    #
    #     print ('\n[test_jpeg_imagedata] jpegfile list = %s' % jpegfile_list)
    #
    #     with self.test_session() as sess:
    #
    #         for filename in jpegfile_list:
    #             print ('[test_jpeg_imagedata] current filename = %s' % filename)
    #
    #
    #             image       = Image.open(filename)
    #             image_numpy = np.array(image).astype(np.uint8)
    #             # plt.imshow(image_numpy)
    #             # plt.show()
    #
    #             with tf.gfile.FastGFile(filename, 'r') as f:
    #                 image_data = f.read()
    #
    #             image_byte  = sess.run([jpeg_tfencode_op],
    #                                    feed_dict={image_in:image_numpy})
    #
    #
    #             # conversion from raw byte
    #             image_numpy2= sess.run([jpeg_tfdecode_op],\
    #                                     feed_dict={image_byte_in:image_data})
    #
    #             # # # conversion from tf encoded jpeg byte
    #             # image_numpy3= sess.run([jpeg_tfdecode_op],\
    #             #                         feed_dict={image_byte_in:image_byte})
    #
    #             plt.imshow(image_numpy2[0])
    #             plt.show()




    def test_read_tfrecords(self):
        '''
            This test checks below:
            - whether tfrecord is correclty read
        '''
        # loading tfrecord filenames from self.data_dir
        train_filename_list = glob(TFRECORD_TEST_DIR + '/train-*.*')

        print('[test_read_tfrecords] data_dir = %s' % TFRECORD_TEST_DIR)
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
        image_byte_input = tf.placeholder(tf.string)

        image_decode_op = tf.image.decode_jpeg(contents=image_byte_input,
                                               channels=DEFAULT_INPUT_CHNUM)

        with self.test_session() as sess:
            sess.run(iterator_train.initializer,
                     feed_dict={filenames: train_filename_list})

            images_byte_op,labels_op = iterator_train.get_next()

            img, la = sess.run([images_byte_op,labels_op])
            image_numpy = sess.run([image_decode_op],feed_dict={image_byte_input:img[0]})

            print ('label_head_x = %s' % la[0])
            print ('label_head_y = %s' % la[1])
            print ('label_head_occ = %s' % la[2])

            plt.imshow(image_numpy[0])
            plt.show()

            # plt.imshow(ima)
            # plt.show()



    # def test_data_loader_tpu(self):
    #     '''
    #         This test checks below:
    #         - whether tfrecord is correclty read
    #     '''
    #     # loading tfrecord filenames from self.data_dir
    #     train_filename_list = glob(TFRECORD_DIR + '/train-*.*')
    #     eval_filename_list  = glob(TFRECORD_DIR + '/eval-*.*')
    #
    #     print('[test_read_tfrecords] data_dir = %s' % TFRECORD_DIR)
    #     print('[test_read_tfrecords] train_filename_list = %s'% train_filename_list)
    #     print('[test_read_tfrecords] eval_filename_list = %s' % eval_filename_list)
    #
    #
    #     filenames = tf.placeholder(tf.string, shape=[None])
    #
    #     dataset_train, dataset_eval = \
    #         [data_loader_tpu.DataSetInput(
    #             is_training=is_training,
    #             data_dir=TFRECORD_DIR,
    #             transpose_input=False,
    #             use_bfloat16=False) for is_training in [True, False]]
    #
    #     dataset_train = dataset_train.input_fn()
    #     dataset_eval  = dataset_eval.input_fn()
    #
    #     iterator_train = dataset_train.make_initializable_iterator()
    #     iterator_eval  = dataset_eval.make_initializable_iterator()
    #
    #
    #
    #     with self.test_session() as sess:
    #         sess.run(iterator_train.initializer,
    #                  feed_dict={filenames: train_filename_list})
    #         sess.run(iterator_eval.initializer,
    #                  feed_dict={filenames: eval_filename_list})
    #
    #         images,labels = iterator_train.get_next()
    #
    #         img, la = sess.run([images,labels])
    #
    #         #
    #         # plt.imshow(ima)
    #         # plt.show()




def dataset_parser(value):
    """Parse an dont be turtle TFrecord from a serialized string Tensor."""
    keys_to_features = {
        'height':
            tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
        'width':
            tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
        'channel':
            tf.FixedLenFeature((), dtype=tf.int64, default_value=3),
        'image':
            tf.FixedLenFeature((), dtype=tf.string, default_value=""),
        'label_head_x':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_head_y':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_head_occ':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_neck_x':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_neck_y':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_neck_occ':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Rshoulder_x':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Rshoulder_y':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Rshoulder_occ':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Lshoulder_x':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Lshoulder_y':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'label_Lshoulder_occ':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'mean':
            tf.VarLenFeature(dtype=tf.float32),
        'std':
            tf.VarLenFeature(dtype=tf.float32),
        "filename":
            tf.FixedLenFeature([], tf.string, default_value="")
    }

    parsed = tf.parse_single_example(serialized =value,
                                     features   =keys_to_features)
    # images
    # image_bytes = tf.reshape(parsed['image'], shape=[])

    image_bytes = parsed['image']

    # labels
    label_head_x        = parsed['label_head_x']
    label_head_y        = parsed['label_head_y']
    label_head_occ      = parsed['label_head_occ']


    label_head_list = [label_head_x,
                       label_head_y,
                       label_head_occ]




    # get the original image shape
    height      = tf.cast(parsed['height']  ,dtype=tf.int32)
    width       = tf.cast(parsed['width']   ,dtype=tf.int32)
    channel     = tf.cast(parsed['channel'] ,dtype=tf.int32)
    mean        = parsed['mean']
    std         = parsed['std']


    return image_bytes, label_head_list



if __name__ == '__main__':
    tf.test.main()

