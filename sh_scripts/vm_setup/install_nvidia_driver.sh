#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm

echo INSTALL PIP
sudo apt-get install python-pip python-numpy swig python-dev python-wheel
sudo apt-get install  python-tk

# 2) GUI + X window off
sudo rm /etc/X11/xorg.conf
sudo service lightdm stop

sudo apt-get --purge remove nvidia-*

echo INSTALL NVIDIA DRIVER
add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
#sudo apt-get install nvidia-384 #recent version at 2017 June // python2
sudo apt-get install nvidia-396 #recent version at 2017 June // python3


echo REBOOT
sudo reboot