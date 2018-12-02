#! /bin/bash

export PROJ_HOME=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/
#export EXPORT_PATH=tfmodules/export/model/run-20181125125403/
export EXPORT_PATH=tfmodules/export/model/run-20181126102146/
export CMD=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/tfmodules

python ${CMD}/gen_tflite_coreml.py  --is-summary=False --import-ckpt-dir=${PROJ_HOME}${EXPORT_PATH}
