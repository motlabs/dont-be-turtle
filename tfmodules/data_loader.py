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

from . import preprocessor

# preprocess_fn 안에 전처리 함수들을 한번에 묶어서 담을 예정
preprocess_fn = preprocessor


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
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(preprocess_fn)
        dataset = dataset.repeat()
        if is_training:
            dataset = dataset.shuffle(buffer_size=(int(len(filenames) * 0.4) + 3 * self.batch_size))
        dataset = dataset.batch(self.batch_size)

        self.iterator = dataset.make_initializable_iterator()
        self.image_stacked, self.label_stacked = self.iterator.get_next()
