# Copyright 2018 Jaewook Kang and JoonHo Lee ({jwkang10, junhoning}@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-#! /usr/bin/env python
'''
    filename: preprocessor.py
    description:
    design doc: https://goo.gl/PTBBVe


    - Author : Junho Lee and Jaewook Kang @ 2018 June

'''

import cv2
import numpy as np


def resize_image(self, np_input_images,weight_resol,height_resol):


# This method resizes the input image arryar, "np_input_images",
#  In this project we have weight:height = 320*480


def augment_image(np_image_array):
    '''
    Scale:
    목적: 사람과 스마트폰의 거리값은 다양할 수 있다는 UX반영
    연산: 이미지를 확대하고 줄인다. (배율을 정할 필요 있음)

    Rotation:
    목적: 손목의 움직임에 따라서 다양한 각도로 촬영 될 수 있다는 UX반영
    연산: 이미지의 각도를 조절한다.
    -15,  -5, +5, +15 (deg)

    Shifting
    목적: 인물은 항상 사진의 중앙에 있지 않기 때문에 그 부분을 반영
    연산: 이미지를 아래 조합으로 linear shift한다
    Shift 정도는 어느정도로 해야하는가?
    좌
    우
    좌상
    좌하
    우상
    우하

    Flip
    목적:
    이미지의 상하반전 : 스마트폰을 거꾸로 거치 할수 있다.
    이미지의 좌우 반전: 데이터 셋이 2배가 된다.
    연산: 이미지의 좌우 반전 and 이미지의 상하 반전
    '''


def annotate_skeleton(np_image_array):
