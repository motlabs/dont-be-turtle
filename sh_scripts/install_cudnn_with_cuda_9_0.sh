#!/bin/bash


# cudnn down link
#wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.4/prod/8.0_20180516/Ubuntu16_04-x64/libcudnn7_7.1.4.18-1%2Bcuda8.0_amd64.deb?Pw--DbIkmF9-Ddyj0vm8cwMG3S5qWfKX8BLEkJugh59ewGUUBkYKguaoPrdq3QD-y1W6B37ADNokJzNLsNska3_Ung6zy1xA19fxXGyUGCg4fAddnBSjtCerrmcE1dPCMfy5dh25_CT2zhjQpbOICx_j6lou6qzOufROICsbO-hshxt6RtO4ojxpYAZzLpghsSEKmw7exvCCbBm8ahGCH5N-UileIhdMHiq5tXw4hPNrjadbQjcgsordinal-virtue-208004
#sudo dpkg -i

sudo cp cuda/lib64/* /usr/local/cuda-9.0/lib64/
sudo cp cuda/include/* /usr/local/cuda-9.0/include/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h

# CuDNN installation check
nvcc --version
cat /proc/driver/nvidia/version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2