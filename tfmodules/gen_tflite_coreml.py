# Copyright 2018 Jaewook Kang (jwkang10@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-


"""Generation of tflite and coreml a dont be turtle model """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import json
from os import listdir

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import tfcoreml as mlmodel_converter

# directory path addition
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import EXPORT_DIR
from path_manager import EXPORT_MODEL_DIR
from path_manager import TF_CNN_MODULE_DIR


# PATH INSERSION
sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,TF_CNN_MODULE_DIR)

sys.path.insert(0,EXPORT_DIR)
sys.path.insert(0,EXPORT_MODEL_DIR)



### models
from model_builder import get_model
from model_config_released  import ModelConfigReleased

model_config = ModelConfigReleased()

class ConvertorToMobileFormat(object):

    def __init__(self,
                 import_model_dir,
                 ckptfilename='model.ckpt',
                 is_summary=False):

        self._ckptfile_name      = ckptfilename
        self._frozen_pb_name     = 'frozen_' + ckptfilename.split('.')[0] + '.pb'
        self._pb_name            = ckptfilename.split('.')[0] + '.pb'
        self._tflite_name       = ckptfilename.split('.')[0] + '.tflite'
        self._mlmodel_name       = ckptfilename.split('.')[0] + '.mlmodel'

        self._is_summary = is_summary
        self._input_graph           =   None

        self._graph_def             =   None
        self._frozen_graph_def      =   None
        self._model_out             =   None
        self._end_points            =   None

        self._saver                 = None # tf.train.Saver()

        self._import_model_dir  = import_model_dir
        self._export_model_dir  = import_model_dir + 'mobile_format/'

        self._input_node_name  = 'model_in'
        self._output_node_name = 'build_network/model/model_out'

        self._input_shape = None
        self._output_shape = None


        if not tf.gfile.Exists(self._export_model_dir):
            tf.gfile.MakeDirs(self._export_model_dir)

        if not tf.gfile.Exists(self._import_model_dir):
            tf.logging.info('[ConvertorToMobileFormat] import_model_dir does not exist!')



    def build_model(self):

        # format NHWC
        self._input_shape = [1,
                       model_config.input_height,
                       model_config.input_width,
                       model_config.input_channel_num]


        # trainable off
        model_config.set_trainable(is_trainable=False)

        # scopes
        build_network_scope = self._output_node_name.split('/')[0]
        model_scope         = self._output_node_name.split('/')[1]

        self._input_graph = tf.Graph()

        with self._input_graph.as_default():

            self._model_in = tf.placeholder(dtype=model_config.dtype,
                                           shape=self._input_shape,
                                           name=self._input_node_name)

            with tf.name_scope(name=build_network_scope):

                self._model_out, _, self._end_points \
                    = get_model(ch_in = self._model_in,
                                model_config = model_config,
                                scope        = model_scope)
                self._init_op   = tf.global_variables_initializer()
                self._saver     = tf.train.Saver(tf.global_variables())

                self._output_shape = self._model_out.get_shape().as_list()

        tf.logging.info('[ConvertorToMobileFormat] model building complete.')

        if self._is_summary == 'True':
            summary_writer  = tf.summary.FileWriter(logdir=self._import_model_dir)
            summary_writer.add_graph(self._input_graph)
            summary_writer.close()




    def convert_and_export(self):

        ckpt_path               = self._import_model_dir + self._ckptfile_name
        export_pb_path          = self._import_model_dir + self._pb_name
        export_frozenpb_path    = self._export_model_dir + self._frozen_pb_name
        export_tflite_path      = self._export_model_dir + self._tflite_name
        export_mlmodel_path     = self._export_model_dir + self._mlmodel_name

        # check_variable_name = 'model/reception/reception/reception_conv7x7_out/weights'
        # chkp.print_tensors_in_checkpoint_file(ckpt_path,
        #                                       tensor_name=check_variable_name,
        #                                       all_tensors= False)

        with tf.Session(graph=self._input_graph) as sess:
            tf.logging.info('------------------------------------------')
            # sess.run(self._init_op)
            self._saver.restore(sess,ckpt_path)

            # export pb
            tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                                 logdir=self._import_model_dir,
                                 name=self._pb_name,
                                 as_text=False)
            tf.logging.info('[ConvertorToMobileFormat] pb is generated.')
            self._graph_def  = sess.graph_def

            # # check collect loading
            # sample_of_restored_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            #                                            scope=check_variable_name)
            # np_restored_var        = sample_of_restored_var[0].eval()

            # tflite generation
            toco = tf.contrib.lite.TocoConverter.from_session(sess=sess,
                                                              input_tensors=[self._model_in],
                                                              output_tensors=[self._model_out])
            tflite_model = toco.convert()


            # frozen pb generation for coreml
            self._frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=self._output_node_name.replace(" ","").split(","))



        with tf.gfile.GFile(export_frozenpb_path,'wb') as f:
            f.write(self._frozen_graph_def.SerializeToString())
            tf.logging.info('[ConvertorToMobileFormat] frozen pb is generated.')

        with tf.gfile.GFile(export_tflite_path,'wb') as f:
            f.write(tflite_model)
            tf.logging.info('[ConvertorToMobileFormat] tflite is generated.')

        # mlmodel (coreml) generation
        mlmodel_converter.convert(tf_model_path=export_frozenpb_path,
                                  mlmodel_path=export_mlmodel_path,
                                  image_input_names=["%s:0" % self._input_node_name],
                                  output_feature_names=["%s:0" % self._output_node_name])
        tf.logging.info('[ConvertorToMobileFormat] mlmodel is generated.')


    def export_shape_in_json(self):

        dict_shape_info= {
            'input_shape': self._input_shape,
            'output_shape': self._output_shape,
            'input_node_name': self._input_node_name,
            'output_node_name': self._output_node_name,
            'keypoints':   ['Head','Nose','Rshoulder','Lshoulder'],
            'dtype':        str(model_config.dtype)

        }

        json_path = self._export_model_dir + 'shape_info.json'
        with open(json_path, 'w') as f:
            json.dump(dict_shape_info,f)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--import-ckpt-dir',
        default=EXPORT_MODEL_DIR,
        nargs='+',
        required=False
    )

    parser.add_argument(
        '--is-summary',
        default=False,
        nargs='+',
        required=False
    )

    args = parser.parse_args()
    filelist = listdir(args.import_ckpt_dir[0])
    filelist_split = filelist[-1].split('.')

    ckptfilename = '.'.join(filelist_split[:2])
    toco = ConvertorToMobileFormat(import_model_dir=args.import_ckpt_dir[0],
                                   ckptfilename=ckptfilename,
                                   is_summary = args.is_summary[0])
    toco.build_model()
    toco.convert_and_export()
    toco.export_shape_in_json()




