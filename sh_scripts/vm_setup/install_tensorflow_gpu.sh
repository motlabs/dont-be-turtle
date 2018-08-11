#! /bin/bash

pip install --user tensorflow-gpu==1.9
pip install --user git+https://github.com/wookayin/tensorflow-plot.git@master
pip install --user tensorpack
pip install --user  Cython
pip install -r  ../../requirement.txt --user

python test_tensorflow_install.py



