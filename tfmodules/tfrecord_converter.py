import os
from glob import glob
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.io import loadmat
import tensorflow as tf


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def to_tfrecords(image_list, label_list, reader, tfrecords_name):
    print("Start converting")
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(path=tfrecords_name, options=options)

    for image_path, label_path in zip(image_list, label_list):
        image, label = reader(image_path)
        filename = os.path.basename(image_path)

        string_set = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'image': _bytes_feature(image.tostring()),
            'label': _bytes_feature(label.tostring()),
            'mean': _float_feature(image.mean().astype(np.float32)),
            'std': _float_feature(image.std().astype(np.float32)),
            'filename': _bytes_feature(str.encode(filename)),
        }))

        writer.write(string_set.SerializeToString())

    writer.close()


def main(train_data_dir, eval_data_dir):
    mat_file = loadmat('mpii_human_pose_v1_u12_1.mat')

    train_data_list = glob(os.path.join(train_data_dir, "image/*.jpg"))
    eval_data_list = glob(os.path.join(train_data_dir, "image/*.jpg"))

    # label 목록을 어디서 봐야 하는지 몰라서 일단 비움

    def reader(path):
        image = Image.open(image_path).astype(np.int64)
        # label = read_label

        return image, label

    to_tfrecords(train_data_list, label_list, reader, 'train_dataset')
    to_tfrecords(eval_data_list, label_list, reader, 'test_dataset')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-data-dir',
        help='training data',
        nargs='+',
        required=True
    )

    parser.add_argument(
        '--eval-data-dir',
        help='evaluation data',
        nargs='+',
        required=True
    )

    args = parser.parse_args()

    main(args.train_data_dir, args.eval_data_dir)
