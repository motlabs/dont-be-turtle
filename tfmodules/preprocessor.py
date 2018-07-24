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
import numpy as np
import sys

from path_manager import TF_MODEL_DIR
sys.path.insert(0,TF_MODEL_DIR)

from model_config  import DEFAULT_INPUT_RESOL
from model_config  import DEFAULT_INPUT_CHNUM
from model_config  import DEFAULT_HG_INOUT_RESOL


IMAGE_SIZE = np.int32(DEFAULT_INPUT_RESOL)

#CROP_PADDING = 32


def _flip(image):
    """Random horizontal image flip."""
    is_flip = tf.less(tf.random_uniform(shape=[1]),0.5)
    is_flip = tf.cast(is_flip,tf.float32)

    image_fliped  = tf.image.flip_left_right(image)
    image         =  image * ( 1.0 - is_flip) + image_fliped * is_flip

    return image, is_flip





def _rotate(image,preproc_config):

    # random angle (rad) geneartion
    min_ang_rad     = preproc_config.MIN_AUGMENT_ROTATE_ANGLE_DEG / 180. * np.pi
    max_ang_rad     = preproc_config.MAX_AUGMENT_ROTATE_ANGLE_DEG / 180. * np.pi


    random_ang_rad  = tf.random_uniform(shape=[1],
                                        minval=min_ang_rad,
                                        maxval=max_ang_rad)

    image = tf.contrib.image.rotate(images=image,
                                    angles=random_ang_rad,
                                    interpolation='BILINEAR')
    return image, random_ang_rad






def preprocess_for_train(image_bytes,use_bfloat16,preproc_config):
    """Preprocesses the given image for evaluation.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """
    # In human pose estmation cropping is not used for possibility to lose body keypoints
    # image = _decode_and_random_crop(image_bytes)

    # here byte to tensor conversion and resize
    # the return has uint8 type
    with tf.name_scope(name='preprocess_for_train', values=[image_bytes]):
        image = tf.image.decode_jpeg(contents=image_bytes,
                                     channels=DEFAULT_INPUT_CHNUM)

        # orignal size to 256
        image = tf.image.resize_bicubic(images=[image],
                                        size=[IMAGE_SIZE, IMAGE_SIZE])[0]
        # augmentation
        if preproc_config.is_flipping:
            image, is_flip = _flip(image=image)
        else:
            is_flip = tf.constant(0.0)

        if preproc_config.is_rotate:
            image, random_ang_rad = _rotate(image=image,
                                            preproc_config=preproc_config)
        else:
            random_ang_rad = tf.constant(0.0)

        # image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, DEFAULT_INPUT_CHNUM])
        # here image value scale is converted to [0,255] to [0.0,1.0]
        image = tf.image.convert_image_dtype(image=image,
                                             dtype=tf.bfloat16 if use_bfloat16 else tf.float32,
                                             saturate=True)

    return image, is_flip, random_ang_rad






def preprocess_for_eval(image_bytes, use_bfloat16):
    """Preprocesses the given image for evaluation.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """
    # In human pose estimation linear shift like augmentation is not used
    # image = _decode_and_center_crop(image_bytes)

    with tf.name_scope(name='preprocess_for_eval', values=[image_bytes]):

        # here byte to tensor conversion and resize
        # the return has uint8 type
        image = tf.image.decode_jpeg(contents=image_bytes,
                                     channels=DEFAULT_INPUT_CHNUM)

        # orignal size to 256
        image = tf.image.resize_bicubic(images=[image],
                                        size =[IMAGE_SIZE,IMAGE_SIZE])[0]

        # image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, DEFAULT_INPUT_CHNUM])

        # here image value scale is converted to [0,255] to [0.0,1.0]
        image = tf.image.convert_image_dtype(image=image,
                                             dtype=tf.bfloat16 if use_bfloat16 else tf.float32,
                                             saturate=True)
    return image





def _heatmap_generator(label_list,
                       image_orig_height,
                       image_orig_width,
                       is_flip,
                       random_ang_rad,
                       use_bfloat16=False,
                       gaussian_ksize=3):

    with tf.name_scope(name='heatmap_generator',values=[label_list,
                                                       image_orig_height,
                                                       image_orig_width,
                                                       is_flip,
                                                       random_ang_rad]):
        x0 = tf.cast(label_list[0], dtype=tf.float32)
        y0 = tf.cast(label_list[1], dtype=tf.float32)

        # reflection of aspect ratio by resizing to DEFAULT_INPUT_RESOL =============
        aspect_ratio_height = DEFAULT_INPUT_RESOL / tf.cast(image_orig_height,dtype=tf.float32)
        aspect_ratio_width  = DEFAULT_INPUT_RESOL / tf.cast(image_orig_width, dtype=tf.float32)

        resized_x0 = x0 * aspect_ratio_width
        resized_y0 = y0 * aspect_ratio_height

        fliped_x0   = (1.0 - is_flip) * resized_x0 + is_flip * (DEFAULT_INPUT_RESOL - resized_x0)
        fliped_y0   = resized_y0


        # reflection of rotation =============
        rotated_x0 = (fliped_x0 - DEFAULT_INPUT_RESOL/2.0) * tf.cos(random_ang_rad) \
                     - (fliped_y0 - DEFAULT_INPUT_RESOL/2.0) * tf.sin(random_ang_rad) \
                     + DEFAULT_INPUT_RESOL/2.0
        rotated_y0 = (fliped_x0 - DEFAULT_INPUT_RESOL/2.0) * tf.sin(random_ang_rad) \
                     + (fliped_y0 - DEFAULT_INPUT_RESOL/2.0) * tf.cos(random_ang_rad) \
                     + DEFAULT_INPUT_RESOL / 2.0

        # resizing by model to  DEFAULT_HG_INOUT_RESOL
        aspect_ratio_by_model = DEFAULT_HG_INOUT_RESOL / DEFAULT_INPUT_RESOL

        heatmap_x0 = rotated_x0 * aspect_ratio_by_model
        heatmap_y0 = rotated_y0 * aspect_ratio_by_model

        # max min bound regularization =============
        heatmap_x0 = tf.minimum(x=heatmap_x0,y=DEFAULT_HG_INOUT_RESOL)
        heatmap_y0 = tf.minimum(x=heatmap_y0,y=DEFAULT_HG_INOUT_RESOL)

        heatmap_x0 = tf.maximum(x=heatmap_x0,y=0.0)
        heatmap_y0 = tf.maximum(x=heatmap_y0,y=0.0)

        # heatmap generation
        label_heatmap = make_gaussian_heatmap(size_h=DEFAULT_HG_INOUT_RESOL,
                                              size_w=DEFAULT_HG_INOUT_RESOL,
                                              fwhm  =gaussian_ksize,
                                              x0    =heatmap_x0,
                                              y0    =heatmap_y0)

        label_heatmap = tf.image.convert_image_dtype(image=label_heatmap,
                                                     dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

    return label_heatmap




def make_gaussian_heatmap(size_h, size_w, x0,y0,fwhm=3):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    if size_h > size_w:
        size = size_h
    else:
        size = size_w

    x = np.arange(0, size, 1, dtype=np.float32)
    y = x[:, np.newaxis]

    heatmap = tf.exp(-4. * tf.log(2.) * ((x - x0) ** 2. + (y - y0) ** 2.) \
                     / fwhm ** 2.)

    return heatmap







def preprocess_image(image_bytes,
                     label_list,
                     image_orig_height,
                     image_orig_width,
                     preproc_config,
                     is_training=False,
                     use_bfloat16=False):

    """Preprocesses the given image.
    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    Returns:
    A preprocessed image `Tensor`.
    """

    with tf.name_scope(name='preprocess_image', values=[image_bytes,
                                                         label_list,
                                                         image_orig_height,
                                                         image_orig_width]):
        # input image preprocessing
        if is_training:
            image,is_flip,random_ang_rad =  preprocess_for_train(image_bytes=image_bytes,
                                                                 use_bfloat16=use_bfloat16,
                                                                 preproc_config=preproc_config)
        else:
            image = preprocess_for_eval(image_bytes=image_bytes,
                                        use_bfloat16=use_bfloat16)
            is_flip         = tf.constant(0.0)
            random_ang_rad  = tf.constant(0.0)


        # label heatmap generation
        label_heatmap_head = _heatmap_generator(label_list      =label_list[0],
                                                image_orig_height=image_orig_height,
                                                image_orig_width =image_orig_width,
                                                is_flip          = is_flip,
                                                random_ang_rad   = random_ang_rad,
                                                use_bfloat16     =use_bfloat16,
                                                gaussian_ksize=preproc_config.heatmap_std)

        label_heatmap_neck = _heatmap_generator(label_list      =label_list[1],
                                                image_orig_height=image_orig_height,
                                                image_orig_width =image_orig_width,
                                                is_flip          = is_flip,
                                                random_ang_rad   = random_ang_rad,
                                                use_bfloat16     =use_bfloat16,
                                                gaussian_ksize=preproc_config.heatmap_std)

        label_heatmap_Rshoulder = _heatmap_generator(label_list =label_list[2],
                                                    image_orig_height=image_orig_height,
                                                    image_orig_width =image_orig_width,
                                                    is_flip          = is_flip,
                                                    random_ang_rad   = random_ang_rad,
                                                    use_bfloat16     =use_bfloat16,
                                                    gaussian_ksize=preproc_config.heatmap_std)

        label_heatmap_Lshoulder = _heatmap_generator(label_list =label_list[3],
                                                    image_orig_height=image_orig_height,
                                                    image_orig_width =image_orig_width,
                                                    is_flip          = is_flip,
                                                    random_ang_rad   = random_ang_rad,
                                                    use_bfloat16     =use_bfloat16,
                                                    gaussian_ksize=preproc_config.heatmap_std)


        label_heatmap           = tf.stack([label_heatmap_head,
                                            label_heatmap_neck,
                                            label_heatmap_Rshoulder,
                                            label_heatmap_Lshoulder],axis=2)

        tf.logging.info('[preprocessor] preprocessing pipeline building complete')
    return image, label_heatmap





## -----------------------------------------------------
# # In human pose estmation cropping is not used
# # for possibility to lose body keypoints
## -----------------------------------------------------
#
# def distorted_bounding_box_crop(image_bytes,
#                                 bbox,
#                                 min_object_covered=0.1,
#                                 aspect_ratio_range=(0.75, 1.33),
#                                 area_range=(0.05, 1.0),
#                                 max_attempts=100,
#                                 scope=None):
#     """Generates cropped_image using one of the bboxes randomly distorted.
#         See `tf.image.sample_distorted_bounding_box` for more documentation.
#         Args:
#         image_bytes: `Tensor` of binary image data.
#         bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
#             where each coordinate is [0, 1) and the coordinates are arranged
#             as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
#             image.
#         min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
#             area of the image must contain at least this fraction of any bounding
#             box supplied.
#         aspect_ratio_range: An optional list of `float`s. The cropped area of the
#             image must have an aspect ratio = width / height within this range.
#         area_range: An optional list of `float`s. The cropped area of the image
#             must contain a fraction of the supplied image within in this range.
#         max_attempts: An optional `int`. Number of attempts at generating a cropped
#             region of the image of the specified constraints. After `max_attempts`
#             failures, return the entire image.
#         scope: Optional `str` for name scope.
#         Returns:
#         (cropped image `Tensor`, distorted bbox `Tensor`).
#
#         https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box
#
#   """
#     with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
#         shape = tf.image.extract_jpeg_shape(image_bytes)
#
#         sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
#             shape,
#             bounding_boxes      =bbox,
#             min_object_covered  =min_object_covered,
#             aspect_ratio_range  =aspect_ratio_range,
#             area_range          =area_range,
#             max_attempts        =max_attempts,
#             use_image_if_no_bounding_boxes=True)
#
#         bbox_begin, bbox_size, _ = sample_distorted_bounding_box
#
#         # Crop the image to the specified bounding box.
#         offset_y, offset_x, _           = tf.unstack(bbox_begin)
#         target_height, target_width, _  = tf.unstack(bbox_size)
#         crop_window                     = tf.stack([offset_y,
#                                                     offset_x,
#                                                     target_height,
#                                                     target_width])
#
#         image = tf.image.decode_and_crop_jpeg(image_bytes,
#                                               crop_window,
#                                               channels=3)
#
#     return image,crop_window
#
#
#
#
# def _at_least_x_are_equal(a, b, x):
#     """At least `x` of `a` and `b` `Tensors` are equal."""
#     match = tf.equal(a, b)
#     match = tf.cast(match, tf.int32)
#     return tf.greater_equal(tf.reduce_sum(match), x)
#
#
#
#
# def _decode_and_random_crop(image_bytes):
#     """Make a random crop of IMAGE_SIZE.
#         For training
#     """
#     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
#
#     image,crop_window = distorted_bounding_box_crop(image_bytes,
#                                                     bbox,
#                                                     min_object_covered=0.1,
#                                                     aspect_ratio_range=(3. / 4, 4. / 3.),
#                                                     area_range=(0.08, 1.0),
#                                                     max_attempts=10,
#                                                     scope=None)
#
#     original_shape  = tf.image.extract_jpeg_shape(image_bytes)
#     bad             = _at_least_x_are_equal(a=original_shape,
#                                             b=tf.shape(image),
#                                             x=3)
#
#     image = tf.cond(bad,
#                     lambda: _decode_and_center_crop(image_bytes),
#                     lambda: tf.image.resize_bicubic([image],  # pylint: disable=g-long-lambda
#                                                     [IMAGE_SIZE, IMAGE_SIZE])[0])
#
#     return image,crop_window
#
#
#
#
#
# def _decode_and_center_crop(image_bytes):
#     """Crops to center of image with padding then scales IMAGE_SIZE.
#         For evaluation
#     """
#
#     shape           = tf.image.extract_jpeg_shape(image_bytes)
#     image_height    = shape[0]
#     image_width     = shape[1]
#
#     # we need to see the below codes
#     padded_center_crop_size = tf.cast(
#         ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
#         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
#         tf.int32)
#
#     offset_height   = ((image_height - padded_center_crop_size) + 1) // 2
#     offset_width    = ((image_width  - padded_center_crop_size) + 1) // 2
#
#     crop_window     = tf.stack([offset_height,
#                                 offset_width,
#                                 padded_center_crop_size,
#                                 padded_center_crop_size])
#
#     image = tf.image.decode_and_crop_jpeg(image_bytes,
#                                           crop_window,
#                                           channels=3)
#
#     image = tf.image.resize_bicubic([image],
#                                     [IMAGE_SIZE, IMAGE_SIZE])[0]
#     return image
#
## -----------------------------------------------------




