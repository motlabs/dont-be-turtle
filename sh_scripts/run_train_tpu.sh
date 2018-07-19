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



echo "  .------------------------."
echo "  |  Hi ! Google Camp 2018 |"
echo "  '------------------------'"
echo "      ^      (\_/)"
echo "      '----- (O.o)"
echo "             (> <)"
OS="$(uname -s)"
OS_X="Darwin"


if [ "$OS" == "$OS_X" ]; then
    export MODEL_BUCKET=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules/export/model
    export DATA_BUCKET=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/dataset/tfrecords
    export SOURCE=~/SourceCodes/dont-be-turtle/tfmodules/trainer_tpu.py
else
    export MODEL_BUCKET=gs://dontbeturtle_ckpt
    export DATA_BUCKET=gs://pose_dataset_tfrecord/tfrecords
    export SOURCE=~/dont-be-turtle/tfmodules/trainer_tpu.py
fi


echo "MODEL_BUCKET="${MODEL_BUCKET}
echo "DATA_BUCKET=" ${DATA_BUCKET}

python ${SOURCE}\
	  --tpu=$USER-tpu \
	  --data_dir=${DATA_BUCKET}/testdataset\
	  --model_dir=${MODEL_BUCKET}