#! /bin/bash

echo CREATE BUCKET
export DATA_BUCKET=gs://pose_dataset_tfrecord
export YOUR_PRJ_NAME=ordinal-virtue-208004
export YOUR_ZONE=us-central1-f
export YOUR_DATASET=/Users/jwkangmacpro2/SourceCodes/dont-be-turtle/dataset/tfrecords

gsutil mb -l ${YOUR_ZONE} -p ${YOUR_PRJ_NAME} ${DATA_BUCKET}

echo COPY DATA TO BUCKET FROM /DATA DIR
gsutil cp -r ${YOUR_DATASET} ${DATA_BUCKET}