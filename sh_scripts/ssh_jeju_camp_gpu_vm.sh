#!/usr/bin/env bash

export VM_NAME=jeju-camp-gpu

gcloud config set compute/zone us-east1-c
gcloud compute ssh ${VM_NAME}