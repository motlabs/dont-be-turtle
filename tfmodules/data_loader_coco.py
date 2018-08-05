# Copyright 2018 Jaewook Kang (jwkang10@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

"""Efficient dont be turtle input pipeline using tf.data.Dataset.
    code ref: https://github.com/edvardHua/PoseEstimationForMobile
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from os.path import join

# for COCO templete
from pycocotools.coco import COCO

from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import COCO_DATALOAD_DIR
from path_manager import COCO_DATASET_BASE_DIR
from path_manager import COCO_REALSET_DIR

sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,COCO_DATALOAD_DIR)
sys.path.insert(0,COCO_REALSET_DIR)

from train_config  import BATCH_SIZE
from train_config  import TRAININGSET_SIZE
from train_config  import PreprocessingConfig
from train_config  import FLAGS

from model_config  import DEFAULT_INPUT_RESOL
from model_config  import DEFAULT_HG_INOUT_RESOL
from model_config  import DEFAULT_INPUT_CHNUM
from model_config  import NUM_OF_KEYPOINTS


# for coco dataset
import dataset_augment
from dataset_prepare import CocoMetadata


DEFAULT_HEIGHT = DEFAULT_INPUT_RESOL
DEFAULT_WIDTH  = DEFAULT_INPUT_RESOL
preproc_config = PreprocessingConfig()


class DataSetInput(object):
    """Generates DataSet input_fn for training or evaluation
        Args:
            is_training: `bool` for whether the input is for training
            data_dir:   `str` for the directory of the training and validation data;
                            if 'null' (the literal string 'null', not None), then construct a null
                            pipeline, consisting of empty images.
            use_bfloat16: If True, use bfloat16 precision; else use float32.
            transpose_input: 'bool' for whether to use the double transpose trick
    """

    def __init__(self, is_training,
                 data_dir,
                 use_bfloat16,
                 transpose_input=True):

        self.image_preprocessing_fn = dataset_augment.preprocess_image
        self.is_training            = is_training
        self.use_bfloat16           = use_bfloat16
        self.data_dir               = data_dir

        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input



    def _set_shapes(self,img, heatmap):
        img.set_shape([BATCH_SIZE,
                       DEFAULT_WIDTH,
                       DEFAULT_HEIGHT,
                       DEFAULT_INPUT_CHNUM])

        heatmap.set_shape([BATCH_SIZE,
                           DEFAULT_HG_INOUT_RESOL,
                           DEFAULT_HG_INOUT_RESOL,
                           NUM_OF_KEYPOINTS])
        return img, heatmap




    def _parse_function(self,imgId, ann=None):
        """
        :param imgId:
        :return:
        """
        global TRAIN_ANNO

        if ann is not None:
            TRAIN_ANNO = ann

        img_meta = TRAIN_ANNO.loadImgs([imgId])[0]
        anno_ids = TRAIN_ANNO.getAnnIds(imgIds=imgId)
        img_anno = TRAIN_ANNO.loadAnns(anno_ids)
        idx = img_meta['id']

        filename_item_list = img_meta['file_name'].split('/')
        filename = filename_item_list[1] +'/' + filename_item_list[2]

        img_path = join(FLAGS.data_dir, filename)

        img_meta_data   = CocoMetadata(idx=idx,
                                       img_path=img_path,
                                       img_meta=img_meta,
                                       annotations=img_anno,
                                       sigma=preproc_config.heatmap_std)

        # print('joint_list = %s' % img_meta_data.joint_list)
        images, labels  = self.image_preprocessing_fn(img_meta_data=img_meta_data,
                                                      preproc_config=preproc_config)
        return images, labels





    def input_fn(self, params=None):
        """Input function which provides a single batch for train or eval.
            Args:
                params: `dict` of parameters passed from the `TPUEstimator`.
                  `params['batch_size']` is always provided and should be used as the
                  effective batch size.
            Returns:
                A `tf.data.Dataset` object.

            doc reference: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
        """
        tf.logging.info('[Input_fn] is_training = %s' % self.is_training)
        json_filename_split = FLAGS.data_dir.split('/')

        if self.is_training:
            json_filename       = json_filename_split[-1] + '_train.json'
        else:
            json_filename       = json_filename_split[-1] + '_valid.json'

        global TRAIN_ANNO

        TRAIN_ANNO      = COCO(join(FLAGS.data_dir,json_filename))
        imgIds          = TRAIN_ANNO.getImgIds()
        dataset         = tf.data.Dataset.from_tensor_slices(imgIds)


        if self.is_training:
            # dataset elementwise shuffling
            dataset = dataset.shuffle(buffer_size=TRAININGSET_SIZE)
            tf.logging.info('[Input_fn] dataset.shuffle()')


        # # Read the data from disk in parallel
        # where cycle_length is the Number of training files to read in parallel.
        multiprocessing_num = 16

        dataset = dataset.map(
            lambda imgId: tuple(
                tf.py_func(
                    func=self._parse_function,
                    inp=[imgId],
                    Tout=[tf.float32, tf.float32]
                )
            ), num_parallel_calls=multiprocessing_num)

        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(self._set_shapes, num_parallel_calls=multiprocessing_num)

        if self.is_training:
            dataset = dataset.repeat()
            tf.logging.info('[Input_fn] dataset.repeat()')

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        tf.logging.info('[Input_fn] dataset pipeline building complete')

        return dataset

