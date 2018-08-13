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
from EmbeddingsParser import EmbeddingsParser
from DataLoader import DataLoader as SciGraphLoader
from TimerCounter import Timer

class BatchifiedEmbeddingsDataLazy():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "interim",
            "neuralnet_training"
    )
    
    ##################################################
    def __del__(self):
        pass
    
    ##################################################
    def __init__(self,use_cuda=True,training=True,data_which="small",embeddings_model="6d50",classes=None,chunk_size=1000):
        if training:
            data_type = "training"
        else:
            data_type = "test"
        
        self.filepath = os.path.join(self.path_persistent,
                                     "-".join([data_type,
                                              "-data-cnn-lazy-",
                                              data_which,
                                              embeddings_model
                                              ])
                                     )
        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        print("Loading word embeddings and parser.")
        self.embeddings_parser = EmbeddingsParser()
        self.embeddings_parser.load_model(embeddings_model)
        self.embeddings_size = EmbeddingsParser.lengths[embeddings_model]
    
        if os.path.isdir(self.filepath):
            print("Loading {} data from disk.".format(data_type))
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"rb") as f:
                self.classes = pickle.load(f)
            file = os.path.join(self.filepath,"labels.pkl")
            with open(file,"rb") as f:
                self.data_labels = pickle.load(f)
            file = os.path.join(self.filepath,"abstracts.pkl")
            with open(file,"rb") as f:
                self.data_abstracts = pickle.load(f)
        
        else:
            print("{} data not on disk.".format(data_type))
            os.mkdir(self.filepath)
            print("Loading and preprocessing {} data.".format(data_type))
            
            print("Loading SciGraph.")
            self.d = SciGraphLoader()
            if training:
                self.d.training_data_for_abstracts(data_which)
            else:
                self.d.test_data_for_abstracts(data_which)
                    
            # initialize labels
            if training:
                self.l = LabelEncoder()
                self.d.data["conferenceseries"] = self.l.fit_transform(self.d.data["conferenceseries"])
                self.classes = np.array(self.l.classes_)
                # append a class for conferences not in the training data
                self.classes = np.append(
                        self.classes,
                        None
                )
            else:
                # update labels with IDs given by LabelEncoder
                for i, c in enumerate(classes):
                    self.d.data.loc[self.d.data["conferenceseries"]==c,"conferenceseries"] = i
                # set label to -1 if not in training data
                self.d.data.loc[pd.to_numeric(self.d.data["conferenceseries"], errors="coerce").isnull(),"conferenceseries"] = len(classes)-1
                self.classes = classes
                
            self.data_labels = np.array(self.d.data["conferenceseries"],dtype="int64")
            
            print("Saving classes.")
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.classes, f)
              
            print("Saving labels.")
            file = os.path.join(self.filepath,"labels.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.data_labels, f)

            print("Preprocessing abstracts.")
            self.data_abstracts = np.array(self.d.data["chapter_abstract"].str.lower())

            print("Saving abstracts.")
            file = os.path.join(self.filepath,"abstracts.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.data_abstracts, f)

            del self.d
            if hasattr(self,"l"):
                del self.l
                
        self.data_size = len(self.data_labels)
    
    ##################################################
    def batchify(self,batch_size,shuffle=True):
        """
        Batchifies the dataset.
        
        Args:
            size (int): number of rows in a batch.
            num_chunks (int): number of chunks to preload into memory.
            shuffle (bool): retrieve randomized batches
        """
        self.batch_size = batch_size
        self.batches = np.arange(self.data_size,step=batch_size)
        
        self.batch_current = 0
        self.batch_max = len(self.batches)
        
        if shuffle:
            order = np.arange(len(self.data_abstracts))
            np.random.shuffle(order)
            self.data_abstracts = self.data_abstracts[order]
            self.data_labels = self.data_labels[order]

    ##################################################
    def next_batch(self):            
        i_from = self.batch_current*self.batch_size
        i_to = i_from + self.batch_size

        # get labels.
        labels = self.data_labels[i_from:i_to]
        labels = torch.tensor(
                labels,
                device=self.device,
                dtype=torch.long
        ).view(len(labels))

        # get abstracts.
        batch = self.embeddings_parser.transform_tensor_to_fixed_size(
                self.data_abstracts[i_from:i_to],
                embeddings_size=self.embeddings_size,
                spatial_size=300
        )
        batch = torch.tensor(
                batch,
                device=self.device,
                dtype=torch.float32
        )
        
        self.batch_current += 1
        
        return batch, labels
    
    ##################################################
    def has_next_batch(self):
        return self.batch_current < self.batch_max
        
    ##################################################
    def num_classes(self):
        return len(self.classes)