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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
from path_manager import TF_MODEL_DIR
sys.path.insert(0,TF_MODEL_DIR)



from model_config  import DEFAULT_INPUT_RESOL
from model_config  import DEFAULT_HG_INOUT_RESOL

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
            tf.FixedLenFeature((),dtype=tf.float32, default_value=0),
        'std':
            tf.FixedLenFeature((),dtype=tf.float32, default_value=0),
        "filename":
            tf.FixedLenFeature([], tf.string, default_value="")
    }

    parsed = tf.parse_single_example(serialized =value,
                                     features   =keys_to_features)
    # images
    # image_bytes = tf.reshape(parsed['image'], shape=[])

    image_bytes = parsed['image']

    # labels
    label_head_x = parsed['label_head_x']
    label_head_y = parsed['label_head_y']
    label_head_occ = parsed['label_head_occ']

    label_neck_x = parsed['label_neck_x']
    label_neck_y = parsed['label_neck_y']
    label_neck_occ = parsed['label_neck_occ']

    label_Rshoulder_x = parsed['label_Rshoulder_x']
    label_Rshoulder_y = parsed['label_Rshoulder_y']
    label_Rshoulder_occ = parsed['label_Rshoulder_occ']

    label_Lshoulder_x = parsed['label_Lshoulder_x']
    label_Lshoulder_y = parsed['label_Lshoulder_y']
    label_Lshoulder_occ = parsed['label_Lshoulder_occ']

    label_head_list = [label_head_x,
                       label_head_y,
                       label_head_occ]

    label_neck_list = [label_neck_x,
                       label_neck_y,
                       label_neck_occ]

    label_Rshoulder_list = [label_Rshoulder_x,
                            label_Rshoulder_y,
                            label_Rshoulder_occ]

    label_Lshoulder_list = [label_Lshoulder_x,
                            label_Lshoulder_y,
                            label_Lshoulder_occ]

    label_list = [label_head_list,
                  label_neck_list,
                  label_Rshoulder_list,
                  label_Lshoulder_list]


    # get the original image shape
    height      = tf.cast(parsed['height']  ,dtype=tf.int32)
    width       = tf.cast(parsed['width']   ,dtype=tf.int32)
    channel     = tf.cast(parsed['channel'] ,dtype=tf.int32)
    mean        = parsed['mean']
    std         = parsed['std']

    shape = [height,width,channel]

    stat=[mean,std]

    return image_bytes, label_list,shape,stat





def _heatmap_generator(label_list,
                       image_orig_height,
                       image_orig_width,
                       is_flip,
                       random_ang_rad,
                       use_bfloat16=False,
                       gaussian_ksize=3):

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
                 - (fliped_x0 - DEFAULT_INPUT_RESOL/2.0) * tf.sin(random_ang_rad) \
                 + DEFAULT_INPUT_RESOL/2.0
    rotated_y0 = (fliped_y0 - DEFAULT_INPUT_RESOL/2.0) * tf.sin(random_ang_rad) \
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

    return label_heatmap,heatmap_x0,heatmap_y0,is_flip,random_ang_rad



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




def argmax_2d(tensor):

    # input format: BxHxWxD
    assert len(tensor.get_shape()) == 4

    with tf.name_scope(name='argmax_2d',values=[tensor]):
        tensor_shape = tensor.get_shape().as_list()

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tensor_shape[0], -1, tensor_shape[3]))

        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.float32)

        # convert indexes into 2D coordinates
        argmax_x = argmax % tensor_shape[2]
        argmax_y = argmax // tensor_shape[2]

    return tf.concat((argmax_x, argmax_y), axis=1)




