#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_load_cpktmeta_tfsummary.py
    
    description: this file is for loading the prototype model of
    pose estimation model and collecting a log summary for 
    tensorboard use in the dont-be-turtle proj.
    
    
    - functions
        - loading model from ckpt and meta files
        - collecting a summary to plotting the model in a tensorboard use.
        
    - Author : jaewook Kang @ 20180613

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from os import getcwd

MODULE_DIR = getcwd() + '/tfmodules'
sys.path.insert(0,MODULE_DIR)

import path_manager
import model_loader as ld
import trainer as tr
import train_config as tr_config

sys.path.insert(0,path_manager.EXPORT_DIR)

model_export_dir        = '/runtrain-20180613-yglee'
import_meta_filename    = model_export_dir + '/net.ckpt.meta'
export_pb_filename      = 'net.pb'

model_loader    = ld.ModelLoader(subdir_and_filename = import_meta_filename)
model_graph_def = model_loader.load_model(clear_devices=True)

# converting to pb / pbdef files
train_config    = tr_config.TrainConfig()

trainer         = tr.Trainer(model_graph=model_graph_def,
                             config=train_config)

trainer.export_graphdef_as_pb(subdir=model_export_dir,filename=export_pb_filename)

