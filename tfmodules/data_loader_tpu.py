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
"""Efficient dont be turtle input pipeline using tf.data.Dataset."""
# code ref: https://github.com/tensorflow/tpu/blob/1fe0a9b8b8df3e2eb370b0ebb2f80eded6a9e2b6/models/official/resnet/imagenet_input.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools

import preprocessing

DEFAULT_HEIGHT  = 256
DEFAULT_WIDTH   = 256

def image_serving_input_fn():
    """Serving input fn for raw images."""

    def _preprocess_image(image_bytes):
        """Preprocess a single raw image."""
        # [[[[the below part]]]]
        # image = resnet_preprocessing.preprocess_image(
        #     image_bytes=image_bytes, is_training=False)

        # null codes
        image = image_bytes
        return image

    image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string)

    # [[[[the below part]]]]
    images = tf.map_fn(_preprocess_image,
                       image_bytes_list,
                       back_prop=False,
                       dtype=tf.float32)

    return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})





class DataSetInput(object):
    """Generates DataSet input_fn for training or evaluation"""

    def __init__(self, is_training,
                 data_dir,
                 use_bfloat16,
                 transpose_input=True):

        self.image_preprocessing_fn = preprocessing.preprocess_image
        self.is_training            = is_training
        self.use_bfloat16           = use_bfloat16
        self.data_dir               = data_dir

        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input





    def set_shapes(self, batch_size,
                   images,
                   labels):
        """Statically set the batch_size dimension."""

        if self.transpose_input:
            images.set_shape(images.get_shape().merge_with(
              tf.TensorShape([None, None, None, batch_size])))

            # [[[[the below part]]]]
            labels.set_shape(labels.get_shape().merge_with(
              tf.TensorShape([batch_size])))
        else:
            images.set_shape(images.get_shape().merge_with(
              tf.TensorShape([batch_size, None, None, None])))

            # [[[[the below part]]]]
            labels.set_shape(labels.get_shape().merge_with(
              tf.TensorShape([batch_size])))

        return images, labels





    def dataset_parser(self, value):
        """Parse an dont be turtle TFrecord from a serialized string Tensor."""
        keys_to_features = {
            "height":
                tf.FixedLenFeature((), tf.int64, default_value=0),
            "width":
                tf.FixedLenFeature((), tf.int64, default_value=0),
            "channel":
                tf.FixedLenFeature((), tf.int64, default_value=3),
            "image":
                tf.FixedLenFeature((), tf.string, default_value=""),
            "label_head":
                tf.FixedLenFeature((), tf.float32, default_value=""),
            "label_neck":
                tf.FixedLenFeature((), tf.float32, default_value=""),
            "label_Rshoulder":
                tf.FixedLenFeature((), tf.float32, default_value=""),
            "label_Lshoulder":
                tf.FixedLenFeature((), tf.float32, default_value=""),
            "mean":
                tf.FixedLenFeature((), tf.float32, default_value=""),
            "std":
                tf.FixedLenFeature((), tf.float32, default_value="")
        }


        parsed = tf.parse_single_example(value, keys_to_features)
        # image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

        # here need to change
        # image = self.image_preprocessing_fn(
        #     image_bytes=image_bytes,
        #     is_training=self.is_training,
        #     use_bfloat16=self.use_bfloat16)

        # get the original image shape
        height      = parsed['height']
        width       = parsed['width']
        channel     = parsed['channel']
        img_shape   = tf.stack([height, width, channel])

        # reshape images
        image   = tf.decode_raw(parsed['image'], tf.int32)
        image   = tf.cast(image, tf.float32)
        image   = tf.reshape(image, img_shape)

        # label
        label_head          = tf.decode_raw(parsed['label_head'],tf.int32)
        label_head          = tf.cast(label_head, tf.float32)

        label_neck          = tf.decode_raw(parsed['label_neck'],tf.int32)
        label_neck          = tf.cast(label_neck, tf.float32)

        label_Rshoulder     = tf.decode_raw(parsed['label_Rshoulder'],tf.int32)
        label_Rshoulder     = tf.cast(label_Rshoulder,tf.float32)

        label_Lshoulder     = tf.decode_raw(parsed['label_Lshoulder'],tf.int32)
        label_Lshoulder     = tf.cast(label_Lshoulder,tf.float32)

        """
            # [[[[the below part]]]]
            # here we should consider about how below four items can be combined
            # label_head,
            # label_neck,
            # label_Rshoulder,
            # label_Lshoulder
        """
        return image, label_head,label_neck,label_Rshoulder,label_Lshoulder





    def input_fn(self, params):
        """Input function which provides a single batch for train or eval.
            Args:
                params: `dict` of parameters passed from the `TPUEstimator`.
                  `params['batch_size']` is always provided and should be used as the
                  effective batch size.
            Returns:
                A `tf.data.Dataset` object.
        """

        if self.data_dir == None:
            tf.logging.info('Using fake input.')
            return self.input_fn_null(params)

        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contrib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern,
                                             shuffle=self.is_training)
        if self.is_training:
            dataset = dataset.repeat()



        def fetch_dataset(filename):
            # [[[[the below part]]]]
            buffer_size = 8 * 1024 * 1024     # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename,
                                              buffer_size=buffer_size)

            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(fetch_dataset,
                                            cycle_length=64,
                                            sloppy=True))
        dataset = dataset.shuffle(1024)

        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
        tf.contrib.data.map_and_batch(self.dataset_parser,
                                        batch_size=batch_size,
                                        num_parallel_batches=8,# 8 == num_cores per host
                                        drop_remainder=True))

        # Transpose for performance on TPU
        if self.transpose_input:
            dataset = dataset.map(
              lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
              num_parallel_calls=8)

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset




    def input_fn_null(self, params):
        """Input function which provides null (black) images."""
        batch_size = params['batch_size']
        dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
        dataset = dataset.prefetch(batch_size)

        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        if self.transpose_input:
          dataset = dataset.map(
              lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
              num_parallel_calls=8)

        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        dataset = dataset.prefetch(32)     # Prefetch overlaps in-feed with training
        tf.logging.info('Input dataset: %s', str(dataset))
        return dataset





    def _get_null_input(self, _):
        null_image = tf.zeros(shape=[DEFAULT_HEIGHT, DEFAULT_WIDTH, 3],
                              dtype=tf.bfloat16 if self.use_bfloat16
                              else tf.float32)
        return (null_image, tf.constant(0, tf.int32))

