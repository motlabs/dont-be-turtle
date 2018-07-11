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

import tensorflow as tf

from hourglass_layer    import get_hourglass_layer
from reception_layer    import get_reception_layer
from supervision_layer  import get_supervision_layer
from output_layer       import get_output_layer

def get_model(ch_in,model_config,scope=None):

    '''

        ch_in (256x256x3) -->
        reception layer (64x64x256) -->
        hourglass layer (64x64x256) -->
        output layer    (64x64x3) -->
        loss
    '''

    net = ch_in
    end_points = {}
    orig_scope = scope
    with tf.variable_scope(name_or_scope=scope,default_name='model',values=[ch_in]) as sc:

        scope = 'reception'
        with tf.variable_scope(name_or_scope=scope, default_name='reception', values=[net]):
            net,end_points_recept, _= get_layer(ch_in       = net,
                                               model_config = model_config.rc_config,
                                               layer_type   = scope)
            end_points.update(end_points_recept)


        scope = 'stacked_hg'
        intermediate_heatmaps = []
        with tf.variable_scope(name_or_scope=scope, default_name='stacked_hg', values=[net]):
            for stacking_index in range(0,model_config.num_of_hgstacking):

                shorcut = net
                # hourglass layer
                net, end_points_hg, _ = get_layer(ch_in          = net,
                                                  model_config   = model_config.hg_config,
                                                  layer_index    = stacking_index,
                                                  layer_type     = 'hourglass')
                end_points.update(end_points_hg)

                # supervision layer
                net, end_points_sv,heatmaps = get_layer(ch_in           = net,
                                                        model_config    = model_config.sv_config,
                                                        layer_index     = stacking_index,
                                                        layer_type      = 'supervision')
                end_points.update(end_points_sv)

                # intermediate heatmap save
                intermediate_heatmaps.append(heatmaps)

                # shortcut sum
                net = tf.add(x=net, y=shorcut,
                             name= 'shortcut_sum' + str(stacking_index))

        # output layer
        scope = 'output'
        with tf.variable_scope(name_or_scope=scope, default_name='output', values=[net]):
            net, end_point_out, _  = get_layer(ch_in            = net,
                                               model_config     = model_config.out_config,
                                               layer_type       = scope)
            end_points.update(end_point_out)


        out = tf.identity(input=net, name= sc.name + '_out')
        end_points[sc.name + '_out'] = out
        end_points[sc.name + '_in'] = ch_in


        return out, intermediate_heatmaps, end_points




def get_layer(ch_in,
              model_config,
              layer_index=0,
              layer_type='hourglass'):


    net = ch_in
    end_points = {}
    heatmaps_out = None


    if layer_type == 'hourglass':
        net, end_points = get_hourglass_layer(ch_in=net,
                                              model_config=model_config,
                                              layer_index=layer_index,
                                              scope=layer_type)
    elif layer_type is 'supervision':
        net, end_points, heatmaps_out = get_supervision_layer(ch_in=net,
                                                              model_config=model_config,
                                                              layer_index=layer_index,
                                                              scope=layer_type)
    elif layer_type is 'reception':
        net, end_points = get_reception_layer(ch_in=net,
                                              model_config=model_config,
                                              scope=layer_type)
    elif layer_type is 'output':
        net, end_points = get_output_layer(ch_in=net,
                                           model_config=model_config,
                                           scope=layer_type)

    return net, end_points, heatmaps_out
