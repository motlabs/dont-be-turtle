#! /bin/bash

#export TRAIN_DATA_DIR=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/dataset/traintest
#export EVAL_DATA_DIR=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/dataset/evaltest

eval "$(pyenv init -)"
pyenv shell tensorflow-anaconda2
echo Convert dataset to TFRecords

python  /Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules/tfrecord_converter.py
