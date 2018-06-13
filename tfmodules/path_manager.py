# -*- coding: utf-8 -*-
# ! /usr/bin/env python
'''
    filename: path_manager.py
    description: this module include all path information on this proj

    - Author : jaewook Kang @ 20180613

'''

from os import getcwd



PROJ_HOME = getcwd()
TF_MODULE_DIR      = PROJ_HOME + '/tfmodules'
EXPORT_DIR         = PROJ_HOME + '/exportfiles'
TENSORBOARD_DIR    = EXPORT_DIR + '/tf_logs'


