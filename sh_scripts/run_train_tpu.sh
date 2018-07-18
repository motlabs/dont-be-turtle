#! /bin/bash

echo            a8888b.
echo           d888888b.
echo           8P"YP"Y88
echo           8|o||o|88
echo           8'    .88
echo           8`._.' Y8.
echo          d/      `8b.
echo         dP   .    Y8b.
echo        d8:'  "  `::88b
echo       d8"         'Y88b
echo      :8P    '      :888
echo       8a.   :     _a88P
echo     ._/"Yaa_:   .| 88P|
echo jgs  \    YP"    `| 8P  `.
echo a:f  /     \.___.d|    .'
echo     `--..__)8888P`._.'


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