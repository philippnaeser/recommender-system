# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:24:14 2018

@author: Steff
"""

import os
import pickle
import sys
import math
import string
import unicodedata

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
    def __init__(self,use_cuda=True,training=True,data_which="small",classes=None):
        if use_cuda:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
            
        if training:
            data_type = "training"
        else:
            data_type = "test"
            
        filepath = os.path.join(self.path_persistent,data_type+"-data-rnn-"+data_which)
        
        if os.path.isdir(filepath):
            print("Loading {} data from disk.")
            file = os.path.join(filepath,"classes.pkl")
            with open(file,"rb") as f:
                self.classes = pickle.load(f)
            file = os.path.join(filepath,"data_labels.pkl")
            with open(file,"rb") as f:
                self.data_labels = pickle.load(f)
            file = os.path.join(filepath,"data_abstracts.pkl")
            with open(file,"rb") as f:
                self.data_abstracts = pickle.load(f)
                
        else:
            print("{} data not on disk.".format(data_type))
            os.mkdir(filepath)
            print("Loading and preprocessing {} data.".format(data_type))
            
            d = SciGraphLoader()
            if training:
                d.training_data(data_which).abstracts()
            else:
                d.test_data(data_which).abstracts()
                
            # drop empty abstracts
            d.data.drop(
                list(d.data[pd.isnull(d.data.chapter_abstract)].index),
                inplace=True
            )
            d.data.reset_index(inplace=True)
            
            self.data_abstracts = d.data["chapter_abstract"].str.lower()
            self.data_labels = d.data["conferenceseries"]
            del d
            
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
                print("updating labels")
                for i, c in enumerate(classes):
                    self.data_labels[self.data_labels==c] = i
                # set label to -1 if not in training data
                self.data_labels[pd.to_numeric(self.data_labels, errors="coerce").isnull()] = len(classes)-1
                self.classes = classes
                
            print("Saving classes.")
            file = os.path.join(filepath,"classes.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.classes, f)
                
            print("Saving labels.")
            file = os.path.join(filepath,"data_labels.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.data_labels, f)
                
            print("Saving abstracts.")
            file = os.path.join(filepath,"data_abstracts.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.data_abstracts, f)
            
        self.letters = string.ascii_letters[0:26] + " .,;'?!=:\"()[]&%+-*/"
        self.letters_size = len(self.letters)
        self.data_size = len(self.data_labels)
        
    ####################################################################
    def _normalize_string(self,s):
        return "".join(
            c for c in unicodedata.normalize("NFD",s)
            if unicodedata.category(c) != "Mn" and c in self.letters
        )
        
    ####################################################################
    def _letterToIndex(self,l):
        return self.letters.find(l)
    
    ####################################################################    
    def _textToTensor(self,text):
        tensor = torch.zeros(
                len(text),
                1,
                self.letters_size,
                dtype=torch.float32,
                device=self.device
        )
        for i, c in enumerate(text):
            tensor[i][0][self._letterToIndex(c)] = 1
        return tensor
    
    ####################################################################
    def shuffle(self):
        order = np.arange(len(self.data_abstracts))
        self.data_abstracts = self.data_abstracts[order]
        self.data_labels = self.data_labels[order]
        self.current_item = 0
        
    ####################################################################
    def has_next(self):
        return self.current_item < self.data_size
        
    ####################################################################
    def next_item(self):
        r = [
                self._textToTensor(self.data_abstracts[self.current_item]),
                torch.LongTensor([self.data_labels[self.current_item]]).view(1)
            ]
        self.current_item += 1
        return r

    ####################################################################
    def num_classes(self):
        return len(self.classes)
    
    ####################################################################
    def num_letters(self):
        return self.letters_size
    

#test = BatchifiedCharactersData(use_cuda=False)