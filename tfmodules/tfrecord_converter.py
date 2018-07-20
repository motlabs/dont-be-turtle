# Copyright 2018 Jaewook Kang and JoonHo Lee ({jwkang10, junhoning}@gmail.com)
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
import os
from glob import glob
from datetime import datetime
import argparse

import numpy as np
import json
from PIL import Image
import tensorflow as tf

from utils import progress_bar

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))





def to_tfrecords(image_list, label_list, reader, tfrecords_name):
    """Converts a dataset to tfrecords."""

    print("Start converting", tfrecords_name)
    # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    options = None
    writer = tf.python_io.TFRecordWriter(path=tfrecords_name, options=options)

    for img_n, (image_path, label_path) in enumerate(zip(image_list, label_list)):
        image, image_jpeg, label = reader(image_path, label_path)
        filename = os.path.basename(image_path)

        string_set = tf.train.Example(
                features=tf.train.Features(
                    feature=
                    {
                        'height'    : _int64_feature(image.shape[0]),
                        'width'     : _int64_feature(image.shape[1]),
                        'channel'   : _int64_feature(image.shape[2]),
                        'image'     : _bytes_feature(image_jpeg),
                        # 'image': _bytes_feature(image.tostring()),

                        '''
                            /* label json format */
                            {
                                "image_path": "/Users/jwkangmacpro2/SourceCodes/dont-be-turtle-pose-annotation-tool/images_for_annotation/lsp_dataset_original/images/im0004.jpg",
                                "head": [550.0049944506104, 386.2047724750277, 0.0],
                                "Rshoulder": [493.74750277469474, 416.89067702552717, 0.0],
                                "Lshoulder": [518.4667036625972, 423.7097669256381, 1.0],
                                "neck": [522.7286348501664, 409.2192008879023, 0.0]
                            }
                            where values of annotation are casted from float32 to int32

                        '''
                        'label_head_x'          : _int64_feature(np.round(label['head'][0]).astype(np.int64)),
                        'label_head_y'          : _int64_feature(np.round(label['head'][1]).astype(np.int64)),
                        'label_head_occ'        : _int64_feature(np.round(label['head'][2]).astype(np.int64)),

                        'label_neck_x'          : _int64_feature(np.round(label['neck'][0]).astype(np.int64)),
                        'label_neck_y'          : _int64_feature(np.round(label['neck'][1]).astype(np.int64)),
                        'label_neck_occ'        : _int64_feature(np.round(label['neck'][2]).astype(np.int64)),

                        'label_Rshoulder_x'     : _int64_feature(np.round(label['Rshoulder'][0]).astype(np.int64)),
                        'label_Rshoulder_y'     : _int64_feature(np.round(label['Rshoulder'][1]).astype(np.int64)),
                        'label_Rshoulder_occ'   : _int64_feature(np.round(label['Rshoulder'][2]).astype(np.int64)),

                        'label_Lshoulder_x'     : _int64_feature(np.round(label['Lshoulder'][0]).astype(np.int64)),
                        'label_Lshoulder_y'     : _int64_feature(np.round(label['Lshoulder'][1]).astype(np.int64)),
                        'label_Lshoulder_occ'   : _int64_feature(np.round(label['Lshoulder'][2]).astype(np.int64)),

                        'mean'              : _float_feature(image.mean().astype(np.float32)),
                        'std'               : _float_feature(image.std().astype(np.float32)),
                        'filename'          : _bytes_feature(filename),
                    }
                )
            )

        writer.write(string_set.SerializeToString())
        progress_bar(len(image_list), img_n + 1, image_path)

    writer.close()


def main(train_dir, eval_dir, out_dir):
    train_data_list = glob(os.path.join(train_dir + 'images/', "*.jp*"))
    eval_data_list = glob(os.path.join(eval_dir   + 'images/', "*.jp*"))


    train_label_list = glob(os.path.join(train_dir + 'labels/', "*.json"))
    eval_label_list = glob(os.path.join(eval_dir   + 'labels/', "*.json"))


    def reader(image_path, label_path):
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)


        with tf.gfile.FastGFile(image_path, 'r') as f:
            image_jpeg = f.read()

        with open(label_path) as label_file:
            label = json.load(label_file)

        return image, image_jpeg, label



    # train_out_path  = os.path.join(out_dir, 'train-dataset.tfrecord.gz')
    # eval_out_path   = os.path.join(out_dir, 'eval-dataset.tfrecord.gz')
    train_out_path  = os.path.join(out_dir, 'train-dataset.tfrecord')
    eval_out_path   = os.path.join(out_dir, 'eval-dataset.tfrecord')


    to_tfrecords(train_data_list,   train_label_list,   reader, train_out_path)
    to_tfrecords(eval_data_list,    eval_label_list,    reader, eval_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-data-dir',
        default = '../dataset/testimages/',
        # default = '../dataset/traintest/lsp/',
        # default='../dataset/train/lsp/',
        help='training data',
        nargs='+',
        required=False
    )

    parser.add_argument(
        '--eval-data-dir',
        default='../dataset/evaltest/collected_data/',
        # default='../dataset/eval/collected_data/',
        help='evaluation data',
        nargs='+',
        required=False
    )

    parser.add_argument(
        '--out-dir',
        default= '../dataset/tfrecords/testimagedataset',
        # default='../dataset/tfrecords/testdataset',
        # default='../dataset/tfrecords/realdataset',
        help='directory of output of data set generated',
        nargs='+',
        required=False
    )

    args = parser.parse_args()
    main(args.train_data_dir, args.eval_data_dir, args.out_dir)
