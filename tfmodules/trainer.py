#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: trainer.py
    description: this is a trainer class in the dont-be-turtle project

    author : Jaewook Kang
    created at 2018 06 13

'''
import argparse
import tensorflow as tf

# from . import path_manager
from data_loader import DataSet


class Trainer(object):
    def __init__(self, train_path, eval_path):
        self.train_path = train_path
        self.eval_path = eval_path

    # def export_graphdef_as_pb(self,subdir,filename):
    #     sess = tf.Session(graph=self._model_graph)
    #     savedir = path_manager.EXPORT_DIR + subdir
    #
    #     # 바이너리로 저장
    #     tf.train.write_graph(sess.graph_def, savedir, filename, as_text=False)
    #     print ("TF graph_def is save in binary at %s" % savedir + '/'+ filename)
    #     tf.train.write_graph(sess.graph_def, savedir, filename+'txt')
    #     print ("TF graph_def is save in txt at %s" % savedir + '/'+ filename+'txt')
    #     print ("---------------------------------------------------------")


    def fit(self, num_epochs, batch_size):
        train_iter = DataSet(batch_size).input_data(self.train_path, is_training=True)
        eval_iter = DataSet(batch_size=1).input_data(self.eval_path, is_training=False)

        # 이 부분에서 에러가 뜨는데. 아직 원인 못 찾음.
        # num_dataset = sum(1 for _ in tf.python_io.tf_record_iterator(self.train_path))
        num_dataset = 10

        train_image, train_label = train_iter.get_next()
        eval_image, eval_label = eval_iter.get_next()

        with tf.Session() as sess:
            sess.run(train_iter.initializer)
            sess.run(eval_iter.initializer)

            epoch = 0
            while True:
                if epoch <= num_epochs:
                    for step in range(num_dataset//batch_size):
                        image, label = sess.run([train_image, train_label])
                        print(image.shape, label.shape)

                else:
                    break


def main(train_path, eval_path, num_epochs, batch_size):
    trainer = Trainer(train_path, eval_path)
    trainer.fit(num_epochs, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-path',
        default='../dataset/train_dataset.tfrecord.gz',
        help='Path of train dataset',
        # nargs='+',
        required=False
    )

    parser.add_argument(
        '--eval-path',
        default='../dataset/eval_dataset.tfrecord.gz',
        help='Path of eval dataset',
        # nargs='+',
        required=False
    )

    parser.add_argument(
        '--num-epochs',
        default=1000,
        help='Number of Epochs for training',
        # nargs='+',
        required=False,
        type=int
    )

    parser.add_argument(
        '--batch-size',
        help='batch size for training',
        default=4,
        required=True,
        type=int
    )

    args = parser.parse_args()

    main(args.train_path, args.eval_path, args.num_epochs, args.batch_size)
