# Copyright 2018 Jaewook Kang and YoungGun Lee ({jwkang10,eofbsdls}@gmail.com)
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
# -*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    function name : heatmap_gen
    description: This function aim to produce a set of heatmaps corresponding to one image input
        - 1) generation of a set of the heatmaps
        - 2) save the heatmaps as .png files to a specified directory path, ./dataset/train/heatmaps
        - 3) a heatmap filename has a form of
            - 'map' + input_image_index + partinfo + '.png'
            - for an input filel, front_turtle_1_s7_0000.jpeg
                - map_0000_neck.png
                - map_0000_head.png
                - map_0000_lsholder.png
                - map_0000_rsholder.png

    ---------------------------------------------------
    data processing flow
    (raw image) -->
    (input: coordinate label) --> (output: heatmap )-->
    (toTFrecord)
    ---------------------------------------------------

    arguments:
        - input_image_filepath: image filepath
            - filename format: location_pose_num_devicename_index.jpeg
                - ex: front_turtle_1_s7_0000.jpeg
        - coor_labels_filepath: coordinate label filepath
            - one images corresponds to total 4 heatmaps
                - head
                - neck
                - lsholder
                - rsholder
        - export_path: path to save heatmap images


    return:
        - a set of output heatmap image filenames as a list of string


'''

def heatmap_gen():


    return



def main(datatype, dataset_path):
    '''
        generation of a set of heatmaps from a set of coordinate label
        description:
            - datatype:
                'train' or 'test'
            - dataset_path
                maybe dataset_path='./dataset' such that
                when datatype ==  'train'
                    - read the label from ./dataset/train/coorlabel
                    - write the heatmap images to ./dataset/train/heatmaps
                when datatype == 'test'
                    - read the label from ./dataset/test/coorlabel
                    - write the heatmap images to ./dataset/test/heatmaps

    '''


if __name__ == "__main__":

    main()



