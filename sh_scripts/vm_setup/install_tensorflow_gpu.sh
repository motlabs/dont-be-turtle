#! /bin/bash

pip install --user opencv-python
pip install --user git+https://github.com/wookayin/tensorflow-plot.git@master
pip install --user tensorpack
pip install --user Cython
pip install --user pycocotools
pip install --user tensorflow-gpu==1.9

python test_tensorflow_install.py



