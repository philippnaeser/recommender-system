# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:24:14 2018

@author: Steff
"""

import os
import pickle
import sys
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from DataLoader import DataLoader as SciGraphLoader
from TimerCounter import Timer

class BatchifiedCharactersData():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "interim",
            "neuralnet_training"
    )
    
    ####################################################################
    def __del__(self):
        pass
    
    ####################################################################
    def __init__(self,training=True,data_which="small",classes=None):
        d = SciGraphLoader()
        if training:
            d.training_data(data_which).abstracts()
        else:
            d.test_data(data_which).abstracts()
            
        self.data_abstracts = d.data["chapter_abstract"]
        self.data_labels = d.data["conferenceseries"]
        
        # initialize labels
        if training:
            self.l = LabelEncoder()
            self.data_labels = self.l.fit_transform(self.data_labels)
            self.classes = np.array(self.l.classes_)
            # append a class for conferences not in the training data
            self.classes = np.append(
                    self.classes,
                    None
            )
        else:
            # update labels with IDs given by LabelEncoder
            for i, c in enumerate(classes):
                self.data_labels[self.data_labels==c] = i
            # set label to -1 if not in training data
            self.data_labels[pd.to_numeric(self.data_labels, errors="coerce").isnull()] = len(classes)-1
            self.classes = classes
        
    
    ####################################################################
    def batchify(self,size,shuffle=True):
        """
        Batchifies the dataset.
        
        Args:
            size (int): number of rows in a batch.
            num_chunks (int): number of chunks to preload into memory.
            shuffle (bool): retrieve randomized batches
        """
        self.batch_size = size

        # initialize batching
        self.batch_current = 0
        self.batch_max = len(self.data_inputs)/self.batch_size
        self.batches = np.arange(len(self.data_abstracts))
        if self.shuffle:
            np.random.shuffle(self.batches)

    ####################################################################
    def next_batch(self):
        #print("Getting batch {}/{}".format(self.batch_current,self.batch_max))
        #timer.tic()
        
        # get (shuffled) indices
        i = self.batch_current * self.batch_size
        indices = self.batches[i:(i+self.batch_size)]

        # get labels
        labels = self.data_labels[indices]
        try:
            labels = torch.cuda.LongTensor(labels).view(len(labels))#,1)
        except TypeError:
            labels = torch.LongTensor(labels).view(len(labels))#,1)
        
        # get abstracts
        #inputs = self.data_inputs[indices]
        inputs = list(self.data_inputs[indices])
        
        # pad inputs to max length
        max_len = max(len(l) for l in inputs)
        #timer.tic() #delme
        for i, inp in enumerate(inputs):
            inputs[i] = np.concatenate((inp,np.zeros(max_len-inp.size)))
        #self.timer_all += timer.toc() #delme
        
        #inputs = torch.cuda.FloatTensor(list(inputs)).unsqueeze(1)
        try:
            inputs = torch.cuda.FloatTensor(inputs).unsqueeze(1)
        except TypeError:
            inputs = torch.FloatTensor(inputs).unsqueeze(1)
        
        self.batch_current += 1
        self.batch_total += 1
        
        #timer.toc()
        return inputs, labels
    
    def has_next_batch(self):
        return (self.batch_current < self.batch_max)
        
    def num_classes(self):
        return len(self.classes)