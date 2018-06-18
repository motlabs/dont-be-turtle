#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: data_loader.py
    description: this module undertakes the below items:
        - Easy delivery of data to the trainer module
        - Easy extraction of input data from image files.
        - Easy extraction of label from image filename.

    design doc: https://goo.gl/PTBBVe
    test data set:
    https://drive.google.com/drive/u/0/folders/18K1-LJ10ABK2TFXtPbLNYa6BI7LsHCUJ

    - Author : Junho Lee and Jaewook Kang @ 2018 June

'''

import numpy as np

class DataSet(object):
    '''
        A class for dataset including (input, label)
    '''
    def __init__(self,input_data=None,label=None,batchsize=1, is_shuffle=False):


        self._np_input_data = input_data
        self._np_label      = label

        self._curr_batch_index   = 0
        self.batchsize      = batchsize
        self._is_shuffle    = is_shuffle


    def reset(self):

        self._np_input_data     = np.empty(shape=[0,0])
        self._np_label          = np.empty(shape=[0,0])
        self._curr_batch_index  = 0


    def set_batchsize(self,batch_size):
        self.batchsize  = batch_size


    def set_dataset(self, input_data,label):

        self._np_input_data = input_data
        self._np_label      = label




    def merge_dataset(self, np_add_dataset):




    def slice_dataset(self, slice_index):




    def get_batch(self):


        # This method returns a batch of a dataset by batch index.
        # ------------------------------


        return np_curr_batch_data, np_curr_batch_label




    def shuffle_dataset(self):

        # This method shuffles an entire dataset
        #-----------------------------------------


        return np_shuffled_dataset




class Dataloader(object):
    '''
         A class downloading, extracting and spliting raw dataset to prepare
         numpy data set.

    '''
    def __init__(self, train_url,test_url,work_dir):
        self._train_url     = train_url
        self._test_url      = test_url
        self._working_dir   = work_dir

        self._np_train_inputdata   = np.empty(shape=[0,0])
        self._np_valid_inputdata   = np.empty(shape=[0,0])
        self._np_test_inputdata    = np.empty(shape=[0,0])

        self._np_train_label       = np.empty(shape=[0,0])
        self._np_valid_label       = np.empty(shape=[0,0])
        self._np_test_label        = np.empty(shape=[0,0])

        self.train_filename = ''
        self.test_filename  = ''

        self.trainset   = DataSet()
        self.validset   = DataSet()
        self.testset    = DataSet()


    def reset(self):


    def download_from_url(self, datatype):


    def extract_data_from_file(self,datatype):



    def extract_lable_from_filename(self,datatype):



    def get_np_data(self,datatype):


    def get_np_label(self,datatype):


    def merge_external_data(self,np_external_data,datatype):









