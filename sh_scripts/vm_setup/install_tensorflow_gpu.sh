#! /bin/bash

pip install --user tensorflow-gpu==1.9
pip install git+https://github.com/wookayin/tensorflow-plot.git@master
pip install --user tensorpack

python test_tensorflow_install.py



