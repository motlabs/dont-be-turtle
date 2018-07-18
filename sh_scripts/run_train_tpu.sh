#! /bin/bash


echo =============================================
echo RUN DONBE-TURTLE-MODEL TRAINING  BY TPU
echo JAEWOOK KANG JEJU GOOGLE CAMP 2018
echo =============================================

export MODEL_BUCKET=gs://dontbeturtle_ckpt
export DATA_BUCKET=gs://pose_dataset_tfrecord/tfrecords

python ~/dont-be-turtle/tfmodules/trainer_tpu.py\
	  --tpu=$USER-tpu \
	  --data_dir=${DATA_BUCKET}\
	  --model_dir=${MODEL_BUCKET}