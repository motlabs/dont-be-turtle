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
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# image processing tools
import cv2

# custom packages
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import COCO_DATALOAD_DIR
from path_manager import COCO_REALSET_DIR

sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,COCO_DATALOAD_DIR)
sys.path.insert(0,COCO_REALSET_DIR)


import data_loader_coco
from train_config import TrainConfig
from train_aux_fn import metric_fn
from train_aux_fn import argmax_2d
from model_config  import DEFAULT_HG_INOUT_RESOL
import tfplot



IMAGE_MAX_VALUE = 255.0
train_config   = TrainConfig()


class DataLoaderTest(tf.test.TestCase):

    def test_data_loader_coco(self):
        '''
            This test checks below:
            - whether tfrecord is correctly read
        '''

        # datadir = TFRECORD_TESTIMAGE_DIR
        datadir = COCO_REALSET_DIR
        # datadir = DATASET_BUCKET
        print('\n---------------------------------------------------------')
        print('[test_data_loader_coco] data_dir = %s' % datadir)

        dataset_train, dataset_valid = \
                [data_loader_coco.DataSetInput(
                    data_dir=datadir,
                    is_training=is_training,
                    transpose_input=False,
                    is_testcode = True,
                    use_bfloat16=False) for is_training in [True,False]]

        dataset = dataset_train
        dataset                 = dataset.input_fn()
        iterator_train          = dataset.make_initializable_iterator()
        feature_op, labels_op   = iterator_train.get_next()
        argmax_2d_top_op            = argmax_2d(tensor=labels_op[:, :, :, 0:1])
        argmax_2d_nose_op           = argmax_2d(tensor=labels_op[:, :, :, 1:2])
        argmax_2d_lshoulder_op      = argmax_2d(tensor=labels_op[:, :, :, 2:3])
        argmax_2d_rshoulder_op      = argmax_2d(tensor=labels_op[:, :, :, 3:4])

        metric_dict_op = metric_fn(labels=labels_op,logits=labels_op,pck_threshold=0.2)
        metric_fn_var  = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope='metric_fn')
        metric_fn_var_init = tf.variables_initializer(metric_fn_var)

        favorite_image_index = 2


        with self.test_session() as sess:
            sess.run(iterator_train.initializer)

            # init variable used in metric_fn_var_init
            sess.run(metric_fn_var_init)

            for n in range(0,30):

                # argmax2d find coordinate of head
                # containing one heatmap
                feature_numpy, labels_numpy, \
                coord_top_numpy,\
                coord_nose_numpy,\
                coord_lshoulder_numpy,\
                coord_rshoulder_numpy,\
                metric_dict   \
                    = sess.run([feature_op,
                                labels_op,
                                argmax_2d_top_op,
                                argmax_2d_nose_op,
                                argmax_2d_lshoulder_op,
                                argmax_2d_rshoulder_op,
                                metric_dict_op])

                # some post processing
                image_sample          = feature_numpy[favorite_image_index,:,:,:]

                print('[test_data_loader_coco] sum of single label heatmap =%s'% \
                      labels_numpy[favorite_image_index, :, :, 0].sum().sum())

                # 256 to 64
                heatmap_size        = int(DEFAULT_HG_INOUT_RESOL)
                image_sample_resized  = cv2.resize(image_sample.astype(np.uint8),
                                                   dsize=(heatmap_size,
                                                          heatmap_size),
                                                   interpolation=cv2.INTER_CUBIC)
                '''
                    marking the annotation
                    # # keypoint_top[0] : x
                    # # keypoint_top[1] : y
                '''
                keypoint_top        = coord_top_numpy[favorite_image_index].astype(np.uint8)
                keypoint_nose       = coord_nose_numpy[favorite_image_index].astype(np.uint8)
                keypoint_lshoulder  = coord_lshoulder_numpy[favorite_image_index].astype(np.uint8)
                keypoint_rshoulder  = coord_rshoulder_numpy[favorite_image_index].astype(np.uint8)

                image_sample_resized[keypoint_top[1],keypoint_top[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_top[1],keypoint_top[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_top[1],keypoint_top[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_nose[1],keypoint_nose[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_nose[1],keypoint_nose[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_nose[1],keypoint_nose[0],2] = IMAGE_MAX_VALUE
                #
                # image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],0] = IMAGE_MAX_VALUE
                # image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],1] = IMAGE_MAX_VALUE
                # image_sample_resized[keypoint_lshoulder[1],keypoint_lshoulder[0],2] = IMAGE_MAX_VALUE

                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],0] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],1] = IMAGE_MAX_VALUE
                image_sample_resized[keypoint_rshoulder[1],keypoint_rshoulder[0],2] = IMAGE_MAX_VALUE



                print ('[test_data_loader_coco] keypoint_top       = (%s,%s)' % (keypoint_top[0],keypoint_top[1]))
                print ('[test_data_loader_coco] keypoint_nose      = (%s,%s)' % (keypoint_nose[0],keypoint_nose[1]))
                print ('[test_data_loader_coco] keypoint_lshoulder = (%s,%s)' % (keypoint_lshoulder[0],keypoint_lshoulder[1]))
                print ('[test_data_loader_coco] keypoint_rshoulder = (%s,%s)' % (keypoint_rshoulder[0],keypoint_rshoulder[1]))

                print (metric_dict)
                print('---------------------------------------------------------\n')

                # print('---------------------------------------------------------')


                # plt.figure(1)
                # plt.imshow(feature_numpy[favorite_image_index].astype(np.uint8))
                # plt.show()

                plt.figure(2)
                plt.imshow(image_sample_resized.astype(np.uint8))
                plt.show()

                #-----------
                labels_top_numpy        = labels_numpy[favorite_image_index, :, :, 0] \
                                          * IMAGE_MAX_VALUE
                labels_nose_numpy       = labels_numpy[favorite_image_index, :, :, 1] \
                                          * IMAGE_MAX_VALUE
                labels_lshoulder_numpy  = labels_numpy[favorite_image_index, :, :, 2] \
                                          * IMAGE_MAX_VALUE
                labels_rshoulder_numpy  = labels_numpy[favorite_image_index, :, :, 3] \
                                          * IMAGE_MAX_VALUE

                ### heatmaps
                if keypoint_top[0] ==0 and keypoint_top[1] ==0:
                    plt.figure(3)
                    plt.imshow(labels_top_numpy.astype(np.uint8))
                    plt.show()

                if keypoint_nose[0] ==0 and keypoint_nose[1] ==0:
                    plt.figure(4)
                    plt.imshow(labels_nose_numpy.astype(np.uint8))
                    plt.show()

                if keypoint_rshoulder[0] ==0 and keypoint_rshoulder[1] ==0:
                    plt.figure(5)
                    plt.imshow(labels_rshoulder_numpy.astype(np.uint8))
                    plt.show()

                if keypoint_lshoulder[0] ==0 and keypoint_lshoulder[1] ==0:
                    plt.figure(6)
                    plt.imshow(labels_lshoulder_numpy.astype(np.uint8))
                    plt.show()

if __name__ == '__main__':
    tf.test.main()

