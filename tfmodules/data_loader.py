# -*- coding: utf-8 -*-
# ! /usr/bin/env python
'''
    filename: data_loader.py
    description: this module undertakes the below items:
        - Easy delivery of data to the trainer module
        - Easy extraction of input data from image files.
        - Easy extraction of label from image filename.

    design doc: https://goo.gl/PTBBVe
    test data set:
    https://drive.google.com/drive/u/0/folders/18K1-LJ10ABK2TFXtPbLNYa6BI7LsHCUJ

    - Author : Junho Lee and Jaewook Kang @ 2018 June

'''
from glob import glob
import numpy as np

import tensorflow as tf

# from . import preprocessor

# preprocess_fn 안에 전처리 함수들을 한번에 묶어서 담을 예정
# preprocess_fn = preprocessor


class DataSet:
    '''
    dataset = Dataset(batch_size=32)
    train_image, train_label, eval_iterator = dataset.input_data(train_files, True)
    eval_image, eval_label, eval_iterator = dataset.input_data(eval_files, False)

    ...

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        image, label = sess.run([self.image_stacked, self.label_stacked])

    '''

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def input_data(self, filenames, is_training):
        compression_type = 'ZLIB' if filenames.split('.')[-1] == 'zlib' else 'GZIP'
        dataset = tf.data.TFRecordDataset([filenames], compression_type=compression_type)

        def parser(record):
            keys_to_features = {
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=3),
                "image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.string, default_value="")
            }
            parsed = tf.parse_single_example(record, keys_to_features)

            # get the original image shape
            height = parsed['height']
            width = parsed['width']
            channel = parsed['channel']
            img_shape = tf.stack([height, width, channel])

            # reshape images
            image = tf.decode_raw(parsed['image'], tf.int32)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, img_shape)

            # temporarily decode label
            label = tf.decode_raw(parsed['label'], tf.int32)
            label = tf.cast(label, tf.float32)

            return image, label

        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        if is_training:
            dataset = dataset.shuffle(buffer_size=(100))  # int(len(filenames) * 0.4) + 3 * self.batch_size)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()

        return iterator
