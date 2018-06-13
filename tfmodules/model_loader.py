'''
    filename: model_loader.py
    description: this module is for loading model and ckpt from
        - model :
            - opt1: .pb
            - opt2: .meta

    author: Jaewook Kang
    created at 2018 6 13
'''

import tensorflow as tf
import sys

# file I/O wrappers without thread locking
from tensorflow.python.platform import gfile
from os import getcwd



EXPORT_FILE_PATH = getcwd()+'/exportfiles'
sys.path.insert(0,EXPORT_FILE_PATH)


class ModelLoader(object):


    def __init__(self,filename):

        # private
        self._filename = filename
        split_filename = filename.split('.')

        # print('[ModelLoader] loading file format is %s' % split_filename[-1])
        if split_filename[-1] == 'pb':
            print ("[ModelLoader] Loading from pb.")
            self._mode='pb'
        elif split_filename[-1] == 'meta':
            print ("[ModelLoader] Loading from meta.")
            self._mode='meta'
        else:
            print ("[ModelLoader] Non-supporting file format.")

            self._mode='notsupp'

        # public
        self.model_graph = tf.Graph()




    def load_model(self,clear_devices=True):

        tf.reset_default_graph()
        model_file_path = EXPORT_FILE_PATH +self._filename
        print ('[ModelLoader] model_file_path = %s' % model_file_path)
        with self.model_graph.as_default():
            if self._mode =='pb':

                with gfile.FastGFile(model_file_path,'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())

                    # Import the graph from "graph_def" into current default graph
                    _ = tf.import_graph_def(graph_def=graph_def,name='')

            elif self._mode == 'meta':
                print('[ModelLoader] Clear device for meta grpah loading = %s' % clear_devices)
                meta_loader = tf.train.import_meta_graph(meta_graph_or_file = model_file_path,
                                                         clear_devices=clear_devices)

                sess = tf.Session(graph= self.model_graph)
                meta_loader.restore(sess,model_file_path[:-5])


        print ("[ModelLoader] Graph loading complete.")

        return self.model_graph


