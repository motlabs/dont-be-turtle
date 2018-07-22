#!/usr/bin/env bash


sudo apt-get install build-essential


# 2) GUI + X window off
sudo rm /etc/X11/xorg.conf
sudo service lightdm stop

sudo apt-get --purge remove nvidia-*

echo INSTALL NVIDIA DRIVER
add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-375 #recent version at 2017 June


echo REBOOT
sudo reboot