import argparse
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import monitors

from .data_loader import DataSet
from ..model_loader import ModelLoader


def model_fn(features, labels, params):
    print('model inference')
    mode = params['mode']

    # TODO: model의 output이 아래에서 나오면 됩니다.
    logit = ModelLoader(features['image'])

    with tf.variable_scope('Prediction'):
        predictions = tf.argmax(logit, axis=-1)

    with tf.variable_scope('Loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions, name='loss_fn')

    with tf.variable_scope('Evaluation'):
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)}

    with tf.variable_scope('Optimization'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   global_step, 30000, 0.5, staircase=True)
        optimizer = params['optimizer'](learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                      predictions=predictions, eval_metric_ops=eval_metric_ops)


def dataset_input_fn(num_batch, filenames):
    iterator = DataSet(num_batch).input_data(filenames, is_training=True)
    features, labels = iterator.get_next()
    return {'image': features}, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluation data',
        nargs='+'
    )
    parser.add_argument(
        '--num-epochs',
        help="""
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.
        """,
        type=int,
        default=0
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=2
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=2
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    parser.add_argument(
        '--train-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=25,
        type=int
    )
    parser.add_argument(
        '--learning-rate',
        help='Initial learning rate',
        default=0.0001,
        type=float
    )
    parser.add_argument(
        '--num-classes',
        help='the number of classes',
        default=5,
        type=int
    )
    parser.add_argument(
        '--mode',
        help='choose one of application mode from: train, eval, infer',
        default='train',
        type=str
    )

    args = parser.parse_args()

    # ------ start script ------ #
    # set verbosity
    tf.logging.set_verbosity(args.verbosity)

    # set param
    model_param = {'learning_rate': args.learning_rate,
                   'optimizer': tf.train.AdamOptimizer,
                   'num_classes': args.num_classes,
                   'mode': args.mode,
                   'batch_size': args.train_batch_size}

    # GPU Configuration
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    # Create the Estimator
    model_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_param,
        model_dir=args.job_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_steps=100,
                                          save_summary_steps=100,
                                          session_config=config))

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: dataset_input_fn(args.train_batch_size, args.eval_files),
        every_n_steps=100,
        eval_steps=20
    )
    hooks = monitors.replace_monitors_with_hooks([validation_monitor], model_classifier)

    # Train the model
    if args.mode == 'train':
        model_classifier.train(input_fn=lambda: dataset_input_fn(args.train_batch_size,
                                                                 args.train_files),
                               steps=args.train_steps,
                               hooks=hooks)

    # Predict using model
    if args.mode == 'eval':
        model_classifier.predict(input_fn=lambda: dataset_input_fn(args.train_batch_size,
                                                                   args.train_files),
                                 predict_keys='classes')
