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
# !/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np

from model_config  import DEFAULT_INPUT_RESOL
from model_config  import NUM_OF_BODY_PART
from model_config  import DEFAULT_LABEL_LENGTH

IMAGE_SIZE = np.int32(DEFAULT_INPUT_RESOL)

CROP_PADDING = 32



def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
        See `tf.image.sample_distorted_bounding_box` for more documentation.
        Args:
        image_bytes: `Tensor` of binary image data.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
            where each coordinate is [0, 1) and the coordinates are arranged
            as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding
            box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional `str` for name scope.
        Returns:
        (cropped image `Tensor`, distorted bbox `Tensor`).

        https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box

  """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
        shape = tf.image.extract_jpeg_shape(image_bytes)

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes      =bbox,
            min_object_covered  =min_object_covered,
            aspect_ratio_range  =aspect_ratio_range,
            area_range          =area_range,
            max_attempts        =max_attempts,
            use_image_if_no_bounding_boxes=True)

        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _           = tf.unstack(bbox_begin)
        target_height, target_width, _  = tf.unstack(bbox_size)
        crop_window                     = tf.stack([offset_y,
                                                    offset_x,
                                                    target_height,
                                                    target_width])

        image = tf.image.decode_and_crop_jpeg(image_bytes,
                                              crop_window,
                                              channels=3)

    return image,crop_window




def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)




def _decode_and_random_crop(image_bytes):
    """Make a random crop of IMAGE_SIZE.
        For training
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    image,crop_window = distorted_bounding_box_crop(image_bytes,
                                                    bbox,
                                                    min_object_covered=0.1,
                                                    aspect_ratio_range=(3. / 4, 4. / 3.),
                                                    area_range=(0.08, 1.0),
                                                    max_attempts=10,
                                                    scope=None)

    original_shape  = tf.image.extract_jpeg_shape(image_bytes)
    bad             = _at_least_x_are_equal(a=original_shape,
                                            b=tf.shape(image),
                                            x=3)

    image = tf.cond(bad,
                    lambda: _decode_and_center_crop(image_bytes),
                    lambda: tf.image.resize_bicubic([image],  # pylint: disable=g-long-lambda
                                                    [IMAGE_SIZE, IMAGE_SIZE])[0])

    return image,crop_window





def _decode_and_center_crop(image_bytes):
    """Crops to center of image with padding then scales IMAGE_SIZE.
        For evaluation
    """

    shape           = tf.image.extract_jpeg_shape(image_bytes)
    image_height    = shape[0]
    image_width     = shape[1]

    # we need to see the below codes
    padded_center_crop_size = tf.cast(
        ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
        tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height   = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width    = ((image_width  - padded_center_crop_size) + 1) // 2

    crop_window     = tf.stack([offset_height,
                                offset_width,
                                padded_center_crop_size,
                                padded_center_crop_size])

    image = tf.image.decode_and_crop_jpeg(image_bytes,
                                          crop_window,
                                          channels=3)

    image = tf.image.resize_bicubic([image],
                                    [IMAGE_SIZE, IMAGE_SIZE])[0]
    return image




def _flip(image):
    """Random horizontal image flip."""
    image = tf.image.random_flip_left_right(image)
    return image


# def _rotation(image):


# under implementation
def _heatmap_generator(label_bytes,use_bfloat16,gaussian_ksize=10):

    labels = tf.decode_raw(bytes=label_bytes,
                          out_type=tf.int32)

    labels.set_shape(labels.get_shape().merge_with
                     (tf.TensorShape([DEFAULT_LABEL_LENGTH])))

    label_len   = labels.get_shape().as_list()[0]

    def make_gaussian(size_h, size_w, fwhm=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        if size_h > size_w:
            size = size_h
        else:
            size = size_w

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        x0 = center[0]
        y0 = center[1]

        temp = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
        return temp[:size_h, :size_w]


    if label_len < (DEFAULT_LABEL_LENGTH - 1):

        label_heatmap_orig = make_gaussian_heatmap()

    #
    #     for i in range(num_of_part):
    #         val_annotation_featureMap_temp = (
    #             make_gaussian(heatmap_height,
    #                           heatmap_width,
    #                           gaussian_ksize,
    #                           (coor_labels[2 * i], coor_labels[2 * i + 1]))).astype(np.float32)
    #
    #         featureMap[:, :, i] = cv2.resize(val_annotation_featureMap_temp,
    #                                          (heatmap_width,heatmap_height),
    #                                          interpolation=cv2.INTER_CUBIC)

    return 0

    # elif length_of_labels == 0:
    #     print('error : coor_labels length is 0. ')
    #     raise ValueError
    # else:
    #     print('error : coor_labels length is odd number.')
    #     raise ValueError




def preprocess_for_train(image_bytes, use_bfloat16):
    """Preprocesses the given image for evaluation.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """
    # here byte to tensor conversion
    image = _decode_and_random_crop(image_bytes)

    image = _flip(image)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image




def preprocess_for_eval(image_bytes, use_bfloat16):
    """Preprocesses the given image for evaluation.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """
    # here byte to tensor conversion
    image = _decode_and_center_crop(image_bytes)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image



def preprocess_image(image_bytes,
                     label_bytes_list,
                     image_orig_height,
                     image_orig_width,
                     is_training=False, use_bfloat16=False):

    """Preprocesses the given image.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """
    # label heatmap generation
    label_heatmap_head = _heatmap_generator(label_bytes=label_bytes_list[0],
                                            image_orig_height=image_orig_height,
                                            image_orig_width =image_orig_width,
                                            use_bfloat16=use_bfloat16,
                                            gaussian_ksize=10)

    # label_heatmap_neck = _heatmap_generator(label_bytes=label_bytes_list[1],
    #                                         use_bfloat16=use_bfloat16,
    #                                         gaussian_ksize=10)
    #
    # label_heatmap_Rshoudler = _heatmap_generator(label_bytes=label_bytes_list[2],
    #                                             use_bfloat16=use_bfloat16,
    #                                             gaussian_ksize=10)
    #
    # label_heatmap_Lshoulder = _heatmap_generator(label_bytes=label_bytes_list[3],
    #                                             use_bfloat16=use_bfloat16,
    #                                             gaussian_ksize=10)
    #
    # label_heatmap           = tf.stack([label_heatmap_head,
    #                                     label_heatmap_neck,
    #                                     label_heatmap_Rshoudler,
    #                                     label_heatmap_Lshoulder])

    # input image preprocessing
    if is_training:
        image =  preprocess_for_train(image_bytes, use_bfloat16)
    else:
        image = preprocess_for_eval(image_bytes, use_bfloat16)


    # return image, label_heatmap
    return image








