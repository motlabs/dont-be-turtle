# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
from os import getcwd
from datetime import datetime
import tensorflow as tf

sys.path.insert(0,getcwd())
sys.path.insert(0,getcwd()+'/..')
sys.path.insert(0,getcwd()+'/../tf-cnn-model')

print ('getcwd() = %s' % getcwd())

from test_layer_util  import create_test_input
from test_layer_util  import get_layer
from test_layer_util  import LayerEndpointName
from test_layer_util  import OutputTestConfig

# where we adopt the NHWC format.

class OutputLayerTest(tf.test.TestCase):



    def test_midpoint_name_shape(self):
        '''
            This test checks below:
            - whether name and shape are correctly set.
        '''

        ch_in_num       = 256
        batch_size      = None
        model_config    = OutputTestConfig()
        scope           = 'unittest'
        TEST_LAYER_NAME = 'output'

        input_shape     = [batch_size,
                           model_config.input_height,
                           model_config.input_width,
                           ch_in_num]

        module_graph = tf.Graph()
        with module_graph.as_default():
            inputs = create_test_input(batchsize    =input_shape[0],
                                       heightsize   =model_config.input_height,
                                       widthsize    =model_config.input_width,
                                       channelnum   =input_shape[3])

            layer_out, mid_points = get_layer(ch_in         = inputs,
                                              model_config  = model_config,
                                              layer_index   = 0,
                                              layer_type    = TEST_LAYER_NAME,
                                              scope         = scope)


        #----------------------------------------------------------
        expected_output_shape   = [batch_size,
                                    model_config.input_height,
                                    model_config.input_width,
                                    model_config.num_of_channels_out]

        expected_midpoint   = LayerEndpointName(layer_type     =TEST_LAYER_NAME,
                                                input_shape     = input_shape,
                                                output_shape    = expected_output_shape)


        expected_input_name = 'unittest0/'+TEST_LAYER_NAME+'_in'
        expected_output_name = 'unittest0/'+TEST_LAYER_NAME+'_out'
        self.assertTrue(expected_input_name in mid_points)
        self.assertTrue(expected_output_name in mid_points)


        print('----------------------------------------------')
        print('[tfTest] run test_midpoint_name_shape()')
        print('[tfTest] midpoint name and shape')
        print('[tfTest] layer_name = %s' % TEST_LAYER_NAME)

        for name, shape in six.iteritems(expected_midpoint.shape_dict):
            print ('%s : shape = %s' % (name,shape))
            self.assertListEqual(mid_points[name].get_shape().as_list(),shape)

        # tensorboard graph summary =============
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        tb_logdir_path = getcwd() + '/tf_logs'
        tb_logdir = "{}/run-{}/".format(tb_logdir_path, now)

        if not tf.gfile.Exists(tb_logdir_path):
            tf.gfile.MakeDirs(tb_logdir_path)


        # summary
        tb_summary_writer = tf.summary.FileWriter(logdir=tb_logdir)
        tb_summary_writer.add_graph(module_graph)
        tb_summary_writer.close()


        # write pbfile of graph_def
        savedir = getcwd() + '/pbfiles/'
        if not tf.gfile.Exists(savedir):
            tf.gfile.MakeDirs(savedir)

        pbfilename      = TEST_LAYER_NAME + '.pb'

        pbtxtfilename   = TEST_LAYER_NAME + '.pbtxt'

        with self.test_session(graph=module_graph) as sess:
            print("TF graph_def is saved in pb at %s" % savedir + pbfilename)
            tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                                 logdir=savedir,
                                 name=pbfilename)
            tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                                 logdir=savedir,
                                 name=pbtxtfilename,as_text=True)





    def test_unknown_batchsize_shape(self):
        '''
            This test check the below case:
            - when a module is built without specifying batch_norm size,
            check whether the model output has a proper batch_size given by an input
        '''

        ch_in_num       = 256
        model_config    = OutputTestConfig()
        scope           = 'unittest'
        TEST_LAYER_NAME = 'output'

        input_shape     = [None,
                           model_config.input_height,
                           model_config.input_width,
                           ch_in_num]

        batch_size      = 1

        inputs = create_test_input(batchsize    =input_shape[0],
                                   heightsize   =input_shape[1],
                                   widthsize    =input_shape[2],
                                   channelnum   =input_shape[3])

        layer_out, mid_points = get_layer(ch_in         = inputs,
                                          model_config  = model_config,
                                          layer_index   = 0,
                                          layer_type    = TEST_LAYER_NAME,
                                          scope         = scope)


        input_shape[0]          = batch_size
        expected_output_shape   = [batch_size,
                                    model_config.input_height,
                                    model_config.input_width,
                                    model_config.num_of_channels_out]

        expected_input_name = 'unittest0/'+TEST_LAYER_NAME+'_in'
        expected_output_name = 'unittest0/'+TEST_LAYER_NAME+'_out'

        self.assertTrue(expected_input_name in mid_points)
        self.assertTrue(expected_output_name in mid_points)

        images = create_test_input(  batchsize=input_shape[0],
                                    heightsize=input_shape[1],
                                     widthsize=input_shape[2],
                                    channelnum=input_shape[3])

        print('----------------------------------------------')
        print('[tfTest] run test_unknown_batchsize_shape()')
        print('[tfTest] midpoint name and shape')
        print('[tfTest] layer_name = %s' % TEST_LAYER_NAME)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(layer_out, {inputs : images.eval()})
            self.assertListEqual(list(output.shape),expected_output_shape)
            print('[TfTest] output shape = %s' % list(output.shape))
            print('[TfTest] expected_output_shape = %s' % expected_output_shape)


if __name__ == '__main__':
    tf.test.main()

