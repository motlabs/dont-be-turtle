#! /bin/bash

echo RUN RESNET TRAINING  BY TPU
export MODEL_BUCKET=gs://dontbeturtle_ckpt
export DATA_BUCKET=gs://pose_dataset_tfrecord/tfrecords

python ~/dont-be-turtle/tfmodules/train_tpu.py\
	  --tpu=$USER-tpu \
	  --data_dir=${DATA_BUCKET}\
	  --model_dir=${MODEL_BUCKET}