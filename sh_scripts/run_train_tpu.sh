#! /bin/bash

OS := $(shell uname)
OS_X := Darwin

echo =============================================
echo RUN DONBE-TURTLE-MODEL TRAINING  BY TPU
echo JAEWOOK KANG JEJU GOOGLE CAMP 2018
echo =============================================

echo "       _                 _         "
echo " _   _| |__  _   _ _ __ | |_ _   _ "
echo "| | | | '_ \| | | | '_ \| __| | | |"
echo "| |_| | |_) | |_| | | | | |_| |_| |"
echo " \__,_|_.__/ \__,_|_| |_|\__|\__,_|"



echo "  .------------------------."
echo "  |  Hi ! Google Camp 2018 |"
echo "  '------------------------'"
echo "      ^      (\_/)"
echo "      '----- (O.o)"
echo "             (> <)"

ifeq ($(OS),$(OS_X))
    export MODEL_BUCKET=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules/export/model
    export DATA_BUCKET=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/dataset/tfrecords
else
    export MODEL_BUCKET=gs://dontbeturtle_ckpt
    export DATA_BUCKET=gs://pose_dataset_tfrecord/tfrecords
endif


echo "MODEL_BUCKET="${MODEL_BUCKET}
echo "DATA_BUCKET=" ${DATA_BUCKET}

python ~/dont-be-turtle/tfmodules/trainer_tpu.py\
	  --tpu=$USER-tpu \
	  --data_dir=${DATA_BUCKET}/testdataset\
	  --model_dir=${MODEL_BUCKET}