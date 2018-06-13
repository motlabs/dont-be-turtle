#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: trainer.py
    description: this is a trainer class in the dont-be-turtle project

    author : Jaewook Kang
    created at 2018 06 13

'''

import tensorflow as tf
import path_manager


class Trainer(object):


    def __init__(self,model_graph,config):

        self._trainconfig   = config
        self._model_graph   = model_graph

    def export_graphdef_as_pb(self,subdir,filename):
        sess = tf.Session(graph=self._model_graph)
        savedir = path_manager.EXPORT_DIR + subdir

        # 바이너리로 저장
        tf.train.write_graph(sess.graph_def, savedir, filename, as_text=False)
        print ("TF graph_def is save in binary at %s" % savedir + '/'+ filename)
        tf.train.write_graph(sess.graph_def, savedir, filename+'txt')
        print ("TF graph_def is save in txt at %s" % savedir + '/'+ filename+'txt')
        print ("---------------------------------------------------------")








