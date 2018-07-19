#! /bin/bash


echo =============================================
echo RUN DONBE-TURTLE-MODEL TRAINING  BY TPU
echo JAEWOOK KANG JEJU GOOGLE CAMP 2018
echo =============================================

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"



echo "  .------------------."
echo "  |  Hi ! Google Camp|"
echo "  '------------------'"
echo "      ^      (\_/)"
echo "      '----- (O.o)"
echo "             (> <)"


export MODEL_BUCKET=gs://dontbeturtle_ckpt
export DATA_BUCKET=gs://pose_dataset_tfrecord/tfrecords

python ~/dont-be-turtle/tfmodules/trainer_tpu.py\
	  --tpu=$USER-tpu \
	  --data_dir=${DATA_BUCKET}\
	  --model_dir=${MODEL_BUCKET}