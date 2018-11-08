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

"""Train a dont be turtle model on TPU.
    under construction since 2018 8
    code reference : https://github.com/tensorflow/tpu/blob/1fe0a9b8b8df3e2eb370b0ebb2f80eded6a9e2b6/models/official/resnet/resnet_main.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os


from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np
from datetime import datetime

# directory path addition
from path_manager import TF_MODULE_DIR
from path_manager import TF_MODEL_DIR
from path_manager import EXPORT_DIR
from path_manager import TF_CNN_MODULE_DIR

from path_manager import TENSORBOARD_BUCKET

# PATH INSERSION
sys.path.insert(0,TF_MODULE_DIR)
sys.path.insert(0,TF_MODEL_DIR)
sys.path.insert(0,TF_CNN_MODULE_DIR)
sys.path.insert(0,EXPORT_DIR)


# custom python packages

### data loader
import data_loader_tpu

### models
from model_builder import get_model
from model_config  import ModelConfig

#### training config
from train_config  import TrainConfig
from train_config  import PreprocessingConfig

from train_config  import LR_SCHEDULE
from train_config  import MEAN_RGB
from train_config  import STDDEV_RGB
from train_config  import FLAGS



from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator


# config instance generation
train_config    = TrainConfig()
model_config    = ModelConfig()
preproc_config  = PreprocessingConfig()



def learning_rate_schedule(current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay.

        The learning rate starts at 0, then it increases linearly per step.
        After 5 epochs we reach the base learning rate (scaled to account
        for batch size).
        After 30, 60 and 80 epochs the learning rate is divided by 10.
        After 90 epochs training stops and the LR is set to 0. This ensures
        that we train for exactly 90 epochs for reproducibility.

        Args:
            current_epoch: `Tensor` for current epoch.

        Returns:
            A scaled `Tensor` for current learning rate.
    """
    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

    decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])

    for mult, start_epoch in LR_SCHEDULE:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)
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
        activation_fn = train_config.activation_fn_out

        if model_config.activation_fn_out == None:
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
                                  predictions=pred_heatmaps[:,:,:,0:1])

        loss_neck       = loss_fn(labels     =label_heatmaps[:,:,:,1:2],
                                  predictions=pred_heatmaps[:,:,:,1:2])

        loss_rshoulder  = loss_fn(labels     =label_heatmaps[:,:,:,2:3],
                                  predictions=pred_heatmaps[:,:,:,2:3])

        loss_lshoulder  = loss_fn(labels     =label_heatmaps[:,:,:,3:4],
                                  predictions=pred_heatmaps[:,:,:,3:4])

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






# def tb_summary_fn_tpu(global_step, loss, mid_loss_list, learning_rate, current_epoch):
def tb_summary_fn_tpu(global_step, loss, learning_rate, current_epoch):

    """Training host call. Creates scalar summaries for training metrics.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`.

    To pass Tensors from the model to the `metric_fn`,
    provide as part of the `host_call`.

    See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `host_call`.

    Args:
      global_step: `Tensor with shape `[batch]` for the global_step
      loss: `Tensor` with shape `[batch]` for the training loss.
      learning_rate: `Tensor` with shape `[batch]` for the learning_rate.
      current_epoch: `Tensor` with shape `[batch]` for the current_epoch.

    Returns:
      List of summary ops to run on the CPU host.
    """

    with tf.name_scope(name='tb_summary',values=[global_step,
                                                 loss,
                                                 # mid_loss_list,
                                                 learning_rate,
                                                 current_epoch]):
        global_step = global_step[0]
        ## create tflog dir
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        tb_logdir_path = TENSORBOARD_BUCKET

        tb_logdir = "{}/run-{}/".format(tb_logdir_path, now)
        tf.logging.info('[model_fn] tf summary at %s' % tb_logdir)

        if not tf.gfile.Exists(tb_logdir_path):
            tf.gfile.MakeDirs(tb_logdir_path)

        with summary.create_file_writer(logdir=tb_logdir).as_default():
            with summary.always_record_summaries():
                summary.scalar('loss', loss[0], step=global_step)

                # for n in range(0,model_config.num_of_hgstacking):
                #     mid_loss_head,\
                #     mid_loss_neck,\
                #     mid_loss_Rshoulder,\
                #     mid_loss_Lshoulder = tf.unstack(mid_loss_list[n])
                #
                #     summary.scalar('mid_loss_head'+str(n), mid_loss_head)
                #     summary.scalar('mid_loss_neck'+str(n), mid_loss_neck)
                #     summary.scalar('mid_loss_Rshoulder'+str(n), mid_loss_Rshoulder)
                #     summary.scalar('mid_loss_Lshoulder'+str(n), mid_loss_Lshoulder)

                summary.scalar('learning_rate', learning_rate[0], step=global_step)
                summary.scalar('current_epoch', current_epoch[0], step=global_step)

    return summary.all_summary_ops()






def model_fn(features,
             labels,
             mode,
             params):
    """
    The model_fn for dontbeturtle model to be used with TPUEstimator.

    Args:
        features:   `Tensor` of batched input images <batchNum x M x M x 3>.
        labels: labels_heatmap_list
        labels =
                        [ [labels_head],
                          [label_neck],
                          [label_rshoulder],
                          [label_lshoulder] ]
                        where has shape <batchNum N x N x 4>

        mode:       one of `tf.estimator.ModeKeys.
                    {
                     - TRAIN (default)  : for weight training ( running forward + backward + metric)
                     - EVAL,            : for validation (running forward + metric)
                     - PREDICT          : for prediction ( running forward only )
                     }`

        Returns:
        A `TPUEstimatorSpec` for the model
    """
    del params # unused

    if isinstance(features, dict):
        features = features['feature']
    if FLAGS.data_format == 'channels_first':
        assert not FLAGS.transpose_input    # channels_first only for GPU
        features = tf.transpose(features, [0, 3, 1, 2])
    if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC




    with tf.name_scope(name='feature_norm',values=[features]):
        # Standardization to the image by zero mean and unit variance.
        features -= tf.constant(MEAN_RGB,   shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

        # set input_shape
        features.set_shape(features.get_shape().merge_with(
            tf.TensorShape([None,
                            model_config.input_height,
                            model_config.input_width,
                            None])))




    # Model building ============================
    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def build_network():
        with tf.name_scope(name='build_network'):
            ''' get model '''
            out_heatmap, mid_heatmap, end_points\
                = get_model(ch_in           = features,
                            model_config    = model_config,
                            scope           = 'model')

            '''specify is_trainable on model '''
            if mode == tf.estimator.ModeKeys.TRAIN:
                model_config.hg_config.is_trainable     = True
                model_config.sv_config.is_trainable     = True
                model_config.rc_config.is_trainable     = True
                model_config.out_config.is_trainable    = True
            elif (mode == tf.estimator.ModeKeys.EVAL) or \
                    (mode == tf.estimator.ModeKeys.PREDICT):
                 model_config.hg_config.is_trainable    = False
                 model_config.sv_config.is_trainable    = False
                 model_config.rc_config.is_trainable    = False
                 model_config.out_config.is_trainable   = False

            tf.logging.info('[model_fn] feature shape=%s' % features.get_shape().as_list())
            tf.logging.info('[model_fn] labels  shape=%s' % labels.get_shape().as_list())
            tf.logging.info('[model_fn] out_heatmap  shape=%s' % out_heatmap.get_shape().as_list())
            tf.logging.info('-----------------------------------------------------------')

            for n in range(0,model_config.num_of_hgstacking):
                tf.logging.info('[model_fn] mid_heatmap%d  shape=%s'
                                % (n,mid_heatmap[n].get_shape().as_list()))

        return out_heatmap, mid_heatmap,end_points


    if FLAGS.precision == 'bfloat16':
        with bfloat16.bfloat16_scope():
            logits_out_heatmap, \
            logits_mid_heatmap, \
            end_points = build_network()
            logits_out_heatmap = tf.cast(logits_out_heatmap, tf.float32)
    else:
        # FLAGS.precision == 'float32':
        logits_out_heatmap, \
        logits_mid_heatmap, \
        end_points = build_network()

    #--------------------------------------------------------
    # mode == prediction case manipulation ===================
    # [[[ here need to change ]]] -----
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     predictions = {
    #
    #         # output format should be clarify here
    #         'pred_head': tf.argmax(logits_heatmap_out[-1,], axis=1),
    #         'conf_head': tf.nn.softmax(logits, name='confidence_head')
    #     }
    #
    #     # if the prediction case return here
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         export_outputs={
    #             'classify': tf.estimator.export.PredictOutput(predictions)
    #         })
    # -----------------------------

    ### output layer ===
    with tf.name_scope(name='out_post_proc', values=[logits_out_heatmap, labels]):
        # heatmap activation of output layer out
        act_out_heatmaps = get_heatmap_activation(logits=logits_out_heatmap,
                                                  scope='out_heatmap')
        # heatmap loss
        total_out_losssum = \
            get_loss_heatmap(pred_heatmaps=act_out_heatmaps,
                             label_heatmaps=labels,
                             scope='out_loss')





    ### middle layer ===
    with tf.name_scope(name='mid_post_proc', values=[logits_mid_heatmap,
                                                     labels]):
        ### supervision layers ===
        act_mid_heatmap_list = []
        total_mid_losssum_list = []
        total_mid_losssum_acc = 0.0

        for stacked_hg_index in range(0, model_config.num_of_hgstacking):
            # heatmap activation of supervision layer out
            act_mid_heatmap_temp = \
                get_heatmap_activation(logits=logits_mid_heatmap[stacked_hg_index],
                                       scope='mid_heatmap_' + str(stacked_hg_index))
            # heatmap loss
            total_mid_losssum_temp = \
                get_loss_heatmap(pred_heatmaps=act_mid_heatmap_temp,
                                 label_heatmaps=labels,
                                 scope='mid_loss_' + str(stacked_hg_index))

            # collect loss and heatmap in list
            act_mid_heatmap_list.append(act_mid_heatmap_temp)
            total_mid_losssum_list.append(total_mid_losssum_temp)
            total_mid_losssum_acc += total_mid_losssum_temp






    ### total loss ===
    with tf.name_scope(name='total_loss',values=[total_out_losssum,
                                                 total_mid_losssum_acc]):
        # Collect weight regularizer loss =====
        loss_regularizer = tf.losses.get_regularization_loss()
        # sum up all losses =====
        loss = total_out_losssum + total_mid_losssum_acc + loss_regularizer



    host_call    = None
    summary_hook = None
    train_op     = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step         = tf.train.get_global_step()
        batchnum_per_epoch  = np.floor(FLAGS.num_train_images / FLAGS.train_batch_size)

        current_epoch       = (tf.cast(global_step, tf.float32) /
                                batchnum_per_epoch)
        learning_rate       = learning_rate_schedule(current_epoch=current_epoch)
        optimizer           = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                        name='RMSprop_opt')



        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
        '''
            # Batch normalization requires UPDATE_OPS to be added as a dependency to
            # the train operation.
            # when training, the moving_mean and moving_variance need to be updated.
        '''
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)





        if FLAGS.is_tensorboard_summary:
            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to
            # [model_config['batch_size']].
            gs_t        = tf.reshape(global_step, [1])
            loss_t      = tf.reshape(loss, [1])

            # mid_loss_list_t = []
            # for n in range(0,model_config.num_of_hgstacking):
            #     mid_loss_list_t[n] = tf.reshape(mid_loss_list[n],[1])

            lr_t = tf.reshape(learning_rate, [1])
            ce_t = tf.reshape(current_epoch, [1])



            if FLAGS.use_tpu:
                # host_call = (tb_summary_fn_tpu, [gs_t, loss_t,mid_loss_list_t, lr_t, ce_t])
                host_call = (tb_summary_fn_tpu, [gs_t, loss_t, lr_t, ce_t])
            else:

                ## create tflog dir
                now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                tb_logdir_path = FLAGS.tflogs_dir

                tb_logdir = "{}/run-{}/".format(tb_logdir_path, now)
                tf.logging.info('[model_fn] tf summary at %s' % tb_logdir)

                if not tf.gfile.Exists(tb_logdir_path):
                    tf.gfile.MakeDirs(tb_logdir_path)
                tf.summary.scalar('loss', loss)

                for n in range(0, model_config.num_of_hgstacking):
                    summary.scalar('mid_loss_head' + str(n), total_mid_losssum_list[n])
                    summary.scalar('mid_loss_neck' + str(n), total_mid_losssum_list[n])
                    summary.scalar('mid_loss_Rshoulder' + str(n), total_mid_losssum_list[n])
                    summary.scalar('mid_loss_Lshoulder' + str(n), total_mid_losssum_list[n])

                    tf.summary.scalar('learning_rate', learning_rate)
                    tf.summary.scalar('current_epoch', current_epoch)

                    tf.logging.info('Create SummarySaveHook.')
                    summary_hook = tf.train.SummarySaverHook(save_steps=FLAGS.summary_step,
                                                             output_dir=tb_logdir,
                                                             summary_op=tf.summary.merge_all())




    if FLAGS.use_tpu:
        # in case of TPUEstimator metric_ops must be in a form of tuple
        metric_ops = (metric_fn, [labels,logits_out_heatmap])
        tfestimator =  tpu_estimator.TPUEstimatorSpec(mode        =mode,
                                                      loss        =loss,
                                                      train_op    =train_op,
                                                      host_call   =host_call,
                                                      eval_metrics=metric_ops)
    else:
        # in case of Estimator metric_ops must be in a form of dictionary
        metric_ops = metric_fn(labels, logits_out_heatmap)
        tfestimator = tf.estimator.EstimatorSpec(mode        =mode,
                                                 loss        =loss,
                                                 train_op    =train_op,
                                                 eval_metric_ops=metric_ops,
                                                 training_hooks = [summary_hook])
    return tfestimator







def main(unused_argv):

    model_config.show_info()
    train_config.show_info()
    preproc_config.show_info()

    ## ckpt dir create
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    curr_model_dir = "{}/run-{}/".format(FLAGS.model_dir, now)

    tf.logging.info('[main] data dir = %s'%FLAGS.data_dir)
    tf.logging.info('[main] model dir = %s'%curr_model_dir)
    tf.logging.info('------------------------')

    if not tf.gfile.Exists(curr_model_dir):
        tf.gfile.MakeDirs(curr_model_dir)

    if FLAGS.use_tpu:
        # for TPU use
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                    FLAGS.tpu,
                    zone=FLAGS.tpu_zone,
                    project=FLAGS.gcp_project)

        # TPU  config
        # config = tpu_config.RunConfig(
        #             cluster                     =tpu_cluster_resolver,
        #             model_dir                   =curr_model_dir,
        #             save_checkpoints_steps      =max(600, FLAGS.iterations_per_loop),
        #             tpu_config                  =tpu_config.TPUConfig(iterations_per_loop         =FLAGS.iterations_per_loop,
        #                                                                 num_shards                  =FLAGS.num_cores,
        #                                                                 per_host_input_for_training =tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

        config = tpu_config.RunConfig(
                    cluster                     =tpu_cluster_resolver,
                    model_dir                   =curr_model_dir,
                    save_checkpoints_steps      =max(600, FLAGS.iterations_per_loop),
                    tpu_config                  =tpu_config.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop))

        tf.logging.info('[main] train_batch_size=%s' % FLAGS.train_batch_size)
        tf.logging.info('[main] eval_batch_size=%s' % FLAGS.eval_batch_size)
        dontbeturtle_estimator = tpu_estimator.TPUEstimator(
                    use_tpu         =FLAGS.use_tpu,
                    model_dir       =curr_model_dir,
                    model_fn        =model_fn,
                    config          =config,
                    train_batch_size=FLAGS.train_batch_size,
                    eval_batch_size =FLAGS.eval_batch_size)

        assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', \
        ('Invalid value for --precision flag; must be bfloat16 or float32.')
        tf.logging.info('Precision: %s', FLAGS.precision)
        use_bfloat16 = (FLAGS.precision == 'bfloat16')
    else:
        # for CPU or GPU use
        config = tf.estimator.RunConfig(
                    model_dir                       =curr_model_dir,
                    tf_random_seed                  =None,
                    save_summary_steps              =FLAGS.summary_step,
                    save_checkpoints_steps          =max(600, FLAGS.iterations_per_loop),
                    session_config                  =None,
                    keep_checkpoint_max             =5,
                    keep_checkpoint_every_n_hours   =10000,
                    log_step_count_steps            =FLAGS.log_step_count_steps,
                    train_distribute                =None)

        dontbeturtle_estimator  = tf.estimator.Estimator(
                    model_dir          = curr_model_dir,
                    model_fn           = model_fn,
                    config             = config,
                    params             = None,
                    warm_start_from    = None)

        use_bfloat16 = False

    '''
    # data loader
    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    '''
    dataset_train, dataset_eval = \
        [data_loader_tpu.DataSetInput(
        is_training     =is_training,
        data_dir        =FLAGS.data_dir,
        transpose_input =FLAGS.transpose_input,
        use_bfloat16    =use_bfloat16) for is_training in [True, False]]



    if FLAGS.mode == 'eval':
        eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

        # Run evaluation when there's a new checkpoint
        for ckpt in evaluation.checkpoints_iterator(
                curr_model_dir, timeout=FLAGS.eval_timeout):
            tf.logging.info('Starting to evaluate.')

            try:
                start_timestamp = time.time()  # This time will include compilation time
                eval_results = dontbeturtle_estimator.evaluate(
                    input_fn        =dataset_eval.input_fn,
                    steps           =eval_steps,
                    checkpoint_path =ckpt)

                elapsed_time = int(time.time() - start_timestamp)
                tf.logging.info('Eval results: %s. Elapsed seconds: %d' %
                                (eval_results, elapsed_time))

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                if current_step >= FLAGS.train_steps:
                    tf.logging.info(
                      'Evaluation finished after training step %d' % current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info(
                    'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

    else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
        current_step = estimator._load_global_step_from_checkpoint_dir(curr_model_dir)  # pylint: disable=protected-access,line-too-long
        batchnum_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size

        tf.logging.info('[main] num_train_images=%s' % FLAGS.num_train_images)
        tf.logging.info('[main] train_batch_size=%s' % FLAGS.train_batch_size)
        tf.logging.info('[main] batchnum_per_epoch=%s' % batchnum_per_epoch)

        tf.logging.info('[main] Training for %d steps (%.2f epochs in total). Current'
                        ' step %d.' % (FLAGS.train_steps,
                                       FLAGS.train_steps / batchnum_per_epoch,
                                       current_step))

        start_timestamp = time.time()  # This time will include compilation time

        if FLAGS.mode == 'train':
            dontbeturtle_estimator.train(input_fn    =dataset_train.input_fn,max_steps   =FLAGS.train_steps)
            tf.logging.info('[main] Training only')

        else:
            assert FLAGS.mode == 'train_and_eval'
            tf.logging.info('[main] Training and Evaluation')

            while current_step < FLAGS.train_steps:
                # Train for up to steps_per_eval number of steps.
                # At the end of training, a checkpoint will be written to --model_dir.
                next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                                      FLAGS.train_steps)
                dontbeturtle_estimator.train(input_fn    =dataset_train.input_fn,max_steps   =next_checkpoint)

                current_step = next_checkpoint

                # Evaluate the model on the most recent model in --model_dir.
                # Since evaluation happens in batches of --eval_batch_size, some images
                # may be consistently excluded modulo the batch size.
                tf.logging.info('Starting to evaluate.')
                eval_results    = dontbeturtle_estimator.evaluate(input_fn=dataset_eval.input_fn,steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
                tf.logging.info('Eval results: %s' % eval_results)

        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                        (FLAGS.train_steps, elapsed_time))

        # if FLAGS.export_dir is not None:
        #     # The guide to serve a exported TensorFlow model is at:
        #     #    https://www.tensorflow.org/serving/serving_basic
        #     tf.logging.info('Starting to export model.')
        #     dontbeturtle_estimator.export_savedmodel(
        #         export_dir_base             =FLAGS.export_dir,
        #         serving_input_receiver_fn   =data_loader_tpu.image_serving_input_fn)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
