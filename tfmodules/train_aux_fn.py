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
#! /usr/bin/env python

import tensorflow as tf


### models
from model_config  import ModelConfig

#### training config
from train_config  import TrainConfig

from train_config  import LR_SCHEDULE
from train_config  import FLAGS
from train_config  import LR_DECAY_RATE

from tensorflow.contrib import summary


# config instance generation
train_config    = TrainConfig()
model_config    = ModelConfig()


def learning_rate_schedule(current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay.

        The learning rate starts at 0, then it increases linearly per step.
        After 5 epochs we reach the base learning rate (scaled to account
        for batch size).
        After 30, 60 and 80 epochs the learning rate is divided by 10.

        Args:
            current_epoch: `Tensor` for current epoch.

        Returns:
            A scaled `Tensor` for current learning rate.
            # Learning rate schedule
                LR_SCHEDULE = [
                    # (multiplier, epoch to start) tuples
                    (1.0, 5), (0.1, 20), (0.01, 60), (0.001, 80)
                ]
    """
    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

    decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])

    for mult, start_epoch in LR_SCHEDULE:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)
    return decay_rate



def learning_rate_exp_decay(current_epoch):

    decay_rate = FLAGS.base_learning_rate  * LR_DECAY_RATE **(current_epoch)
    return decay_rate




def argmax_2d(tensor):

    # input format: BxHxWxD
    assert len(tensor.get_shape()) == 4

    with tf.name_scope(name='argmax_2d',values=[tensor]):
        tensor_shape = tensor.get_shape().as_list()

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tensor_shape[0], -1, tensor_shape[3]))

        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.float32)

        # convert indexes into 2D coordinates
        argmax_x = argmax % tensor_shape[2]
        argmax_y = argmax // tensor_shape[2]

    return tf.concat((argmax_x, argmax_y), axis=1)





def get_heatmap_activation(logits,scope=None):
    '''
        get_heatmap_activation()

        :param logits: NxNx4 logits before activation
        :param scope: scope
        :return: NxNx4 heatmaps

        where we use tf.metrics.mean_squared_error.
        for detail plz see
        https://www.tensorflow.org/api_docs/python/tf/metrics/mean_squared_error

        written by Jaewook Kang July 2018
    '''
    with tf.name_scope(name=scope, default_name='heatmap_act',values=[logits]):

        # ### 1) split logit to head, neck, Rshoulder, Lshoulder
        # logits_heatmap_head, \
        # logits_heatmap_neck, \
        # logits_heatmap_rshoulder, \
        # logits_heatmap_lshoulder = tf.split(logits,
        #                                     num_or_size_splits=model_config.num_of_labels,
        #                                     axis=3)
        ### 2) activation
        activation_fn = train_config.activation_fn_pose

        if train_config.activation_fn_pose == None:
            ''' linear activation case'''
            act_heatmap_head        = logits[:,:,:,0:1]
            act_heatmap_neck        = logits[:,:,:,1:2]
            act_heatmap_rshoulder   = logits[:,:,:,2:3]
            act_heatmap_lshoulder   = logits[:,:,:,3:4]
        else:
            act_heatmap_head      = activation_fn(logits[:,:,:,0:1],
                                                  name='act_head')
            act_heatmap_neck      = activation_fn(logits[:,:,:,1:2],
                                                  name='act_neck')
            act_heatmap_rshoulder = activation_fn(logits[:,:,:,2:3],
                                                  name='act_rshoulder')
            act_heatmap_lshoulder = activation_fn(logits[:,:,:,3:4],
                                                  name='act_lshoulder')

        act_heatmaps = tf.concat([act_heatmap_head, \
                                 act_heatmap_neck, \
                                 act_heatmap_rshoulder, \
                                 act_heatmap_lshoulder],axis=3)
    return act_heatmaps





def get_loss_heatmap(pred_heatmaps,
                     label_heatmaps,
                     scope=None):
    '''
        get_loss_heatmap()

        :param pred_heatmap_list:
            predicted heatmaps <NxNx4> given by model

        :param label_heatmap_list:
            the ground true heatmaps <NxNx4> given by training data

        :param scope: scope
        :return:
            - total_losssum: the sum of all channel losses
            - loss_tensor: loss tensor of the four channels

        written by Jaewook Kang 2018
    '''

    with tf.name_scope(name=scope,default_name='loss_heatmap'):

        # ### 1) split logit to head, neck, Rshoulder, Lshoulder
        # pred_heatmap_head, \
        # pred_heatmap_neck, \
        # pred_heatmap_rshoulder, \
        # pred_heatmap_lshoulder = tf.split(pred_heatmaps,
        #                                     num_or_size_splits=model_config.num_of_labels,
        #                                     axis=3)
        # label_heatmap_head, \
        # label_heatmap_neck, \
        # label_heatmap_rshoulder, \
        # label_heatmap_lshoulder = tf.split(label_heatmaps,
        #                                     num_or_size_splits=model_config.num_of_labels,
        #                                     axis=3)
        #

        ### 3) get loss function of each part


        loss_fn         = train_config.heatmap_loss_fn
        loss_head       = loss_fn(labels     =label_heatmaps[:,:,:,0:1],
                                  predictions=pred_heatmaps[:,:,:,0:1]) \
                          / tf.reduce_mean(label_heatmaps[:,:,:,0:1])

        loss_neck       = loss_fn(labels     =label_heatmaps[:,:,:,1:2],
                                  predictions=pred_heatmaps[:,:,:,1:2]) \
                          / tf.reduce_mean(label_heatmaps[:, :, :, 1:2])

        loss_rshoulder  = loss_fn(labels     =label_heatmaps[:,:,:,2:3],
                                  predictions=pred_heatmaps[:,:,:,2:3]) \
                          / tf.reduce_mean(label_heatmaps[:, :, :, 2:3])

        loss_lshoulder  = loss_fn(labels     =label_heatmaps[:,:,:,3:4],
                                  predictions=pred_heatmaps[:,:,:,3:4]) \
                          / tf.reduce_mean(label_heatmaps[:, :, :, 3:4])

        # loss_tensor = tf.stack([loss_head, loss_neck, loss_rshoulder, loss_lshoulder])
        total_losssum = loss_head + loss_neck + loss_rshoulder + loss_lshoulder

    return total_losssum





def metric_fn(labels, logits):
    """Evaluation metric function. Evaluates accuracy.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the model
    to the `metric_fn`, provide as part of the `eval_metrics`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `eval_metrics`.

    Args:
    labels: `Tensor` of labels_heatmap_list
    logits: `Tensor` of logits_heatmap_list

    Returns:
    A dict of the metrics to return from evaluation.
    """

    with tf.name_scope('metric_fn',values=[labels, logits]):
        # logits_head,\
        # logits_neck,\
        # logits_rshoulder,\
        # logits_lshoulder = tf.split(logits,
        #                             num_or_size_splits=model_config.num_of_labels,
        #                             axis=3)
        #
        # label_head, \
        # label_neck, \
        # label_rshoulder, \
        # label_lshoulder = tf.split(labels,
        #                             num_or_size_splits=model_config.num_of_labels,
        #                             axis=3)

        # get predicted coordinate
        pred_head_xy       = argmax_2d(logits[:,:,:,0:1])
        pred_neck_xy       = argmax_2d(logits[:,:,:,1:2])
        pred_rshoulder_xy  = argmax_2d(logits[:,:,:,2:3])
        pred_lshoulder_xy  = argmax_2d(logits[:,:,:,3:4])

        label_head_xy      = argmax_2d(labels[:,:,:,0:1])
        label_neck_xy      = argmax_2d(labels[:,:,:,1:2])
        label_rshoulder_xy = argmax_2d(labels[:,:,:,2:3])
        label_lshoulder_xy = argmax_2d(labels[:,:,:,3:4])

        # error distance measure
        metric_err_fn                 = train_config.metric_fn
        head_neck_dist, update_op     = metric_err_fn(labels=label_head_xy,
                                                      predictions=label_neck_xy)

        errdist_head,update_op_errdist_head             = metric_err_fn(labels=label_head_xy,
                                                                        predictions=pred_head_xy)
        errdist_neck,update_op_errdist_neck             = metric_err_fn(labels=label_neck_xy,
                                                                        predictions= pred_neck_xy)
        errdist_rshoulder, update_op_errdist_rshoulder  = metric_err_fn(labels=label_rshoulder_xy,
                                                                        predictions= pred_rshoulder_xy)
        errdist_lshoulder, update_op_errdist_lshoulder  = metric_err_fn(labels=label_lshoulder_xy,
                                                                        predictions= pred_lshoulder_xy)
        # percentage of correct keypoints
        total_errdist = (errdist_head +\
                        errdist_neck +\
                        errdist_rshoulder +\
                        errdist_lshoulder) / head_neck_dist

        pck =            tf.metrics.percentage_below(values=total_errdist,
                                                   threshold=FLAGS.pck_threshold,
                                                   name=    'pck_' + str(FLAGS.pck_threshold))

        # form a dictionary
        metric_dict = {
                            'head_neck_dist' : head_neck_dist,
                            'errdist_head': (errdist_head      / head_neck_dist,
                                             update_op_errdist_head),
                            'errdist_neck': (errdist_neck      / head_neck_dist,
                                             update_op_errdist_neck),
                            'errdist_rshoulder': (errdist_rshoulder / head_neck_dist,
                                                    update_op_errdist_rshoulder),
                            'errdist_lshoulder': (errdist_lshoulder / head_neck_dist,
                                                    update_op_errdist_lshoulder),
                            'pck': pck
                        }

    return metric_dict


