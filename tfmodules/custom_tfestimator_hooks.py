
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
"""Custom SessionRunHook classes for the dontbe turtle proj."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from path_manager import TFLITE_CUSTOM_TOCO_DIR
sys.path.insert(0,TFLITE_CUSTOM_TOCO_DIR)


from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.contrib.lite    import TocoConverter
from tflite_convertor           import TFliteConvertor as CustomTocoConverter



class TfliteSaverHook(session_run_hook.SessionRunHook):
    '''Save Tflite file from session
        - Use example:

        tflite_export_hook = TfliteSaverHook(input_tensor       = features,
                                             output_tensor      = logits_out_heatmap,
                                             output_node_name   ='build_network/model/model_out',
                                             savedir            =FLAGS.model_dir)

    '''


    def __init__(self,
                 input_tensors,
                 output_tensors,
                 output_node_name,
                 ckpt_filename='model.ckpt',
                 pb_filename='graph.pbtxt',
                 model_name='model',
                 savedir='./'):

        ''' Init a TfliteSaverHook

        Args:
            input_tensor: input tensor of the model which wish to convert to tflite, whose shape is in a NHWC format.
            output_tensor: output tensor of the model which wishj to convert to tflite, whose shape is in a NHWC format.
            tflitename: the filename of the tflite file.
            tflitedir : the directory path where the tflite file is saved.

        '''

        logging.info("[TfliteSaverHook] Create TfliteSaverHook")

        self._output_node_name= output_node_name
        self._ckpt_filename   = ckpt_filename
        self._pb_filename     = pb_filename

        self._save_dir      = savedir
        self._tflitename    = model_name + '.tflite'
        self._frozenpbname  = 'frozen_' + model_name + '.pb'
        self._input_tensor  = input_tensors
        self._output_tensor = output_tensors


    def begin(self):
        if not gfile.Exists(self._tflite_dir):
            gfile.MakeDirs(self._tflite_dir)


    def after_create_session(self, session, coord):
        pass



    def before_run(self, run_context):
        pass



    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
        pass



    def end(self, session):

        self.convert_to_frozen_pb()
        toco    = TocoConverter.from_session(sess            = session,
                                            input_tensors   = [self._input_tensor],
                                            output_tensors  = [self._output_tensor])

        tflite_model    = toco.convert()
        open(self._save_dir + self._tflitename, 'wb').write(tflite_model)
        logging.info("[TfliteSaverHook] Tflite successfully created.")



    def convert_to_frozen_pb(self):
        tflite_convertor    = CustomTocoConverter()

        # converting to frozen graph
        tflite_convertor.set_config_for_frozen_graph(input_dir_path=self._save_dir,
                                                     input_pb_name=self._pb_filename,
                                                     input_ckpt_name=self._ckpt_filename,
                                                     output_dir_path=self._save_dir,
                                                     output_node_names=self._output_node_name)
        tflite_convertor.convert_to_frozen_graph()
        logging.info('[TfliteSaverHook] Frozen graph is successfully created.')



