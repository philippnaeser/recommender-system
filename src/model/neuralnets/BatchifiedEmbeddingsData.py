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

class BatchifiedEmbeddingsData():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "interim",
            "neuralnet_training"
    )
    
    def __del__(self):
        pass
    
    def __init__(self,use_cuda=True,training=True,data_which="small",glove_model="6d50",classes=None,chunk_size=1000):
        if training:
            data_type = "training"
        else:
            data_type = "test"
        
        self.filepath = os.path.join(self.path_persistent,data_type+"-data-cnn-"+data_which+glove_model)
        self.timer = Timer()
        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        if os.path.isdir(self.filepath):
            print("Loading {} data from disk. Lazy loading inputs.".format(data_type))
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"rb") as f:
                self.classes = pickle.load(f)
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"rb") as f:
                self.chunks_max, self.chunk_size, self.size = pickle.load(f)
        
        else:
            print("{} data not on disk.".format(data_type))
            os.mkdir(self.filepath)
            print("Loading and preprocessing {} data.".format(data_type))
            
            self.timer.tic()
            print("Loading SciGraph.")
            self.d = SciGraphLoader()
            if training:
                self.d.training_data(data_which).abstracts()
            else:
                self.d.test_data(data_which).abstracts()
            self.timer.toc()
            
            # drop empty abstracts
            self.d.data.drop(
                list(self.d.data[pd.isnull(self.d.data.chapter_abstract)].index),
                inplace=True
            )
            self.d.data.reset_index(inplace=True)
                    
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
                
            data_labels = np.array(self.d.data["conferenceseries"],dtype="int64")
            
            print("Saving classes.")
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.classes, f)
            
            print("Loading word embeddings and parser.")
            self.timer.tic()
            # load GloVe model
            self.glove = EmbeddingsParser()
            self.glove.load_model(glove_model)
            self.timer.toc()
            
            print("Preprocessing abstracts.")
            self.timer.tic()
            inputs = list(self.d.data["chapter_abstract"].str.lower())
            self.timer.toc()

            chunks_max = math.ceil(len(inputs)/chunk_size)
            print("Transforming and saving abstracts: {} chunks.".format(chunks_max))
            self.timer.tic()
            self.timer.set_counter(chunks_max,max=10)
            for chunk in np.arange(chunks_max):
                vectors = self.glove.transform_vectors(inputs[chunk*chunk_size:(chunk+1)*chunk_size])
                file = os.path.join(self.filepath,"data."+str(chunk))
                with open(file,"wb") as f:
                    # save labels with abstracts
                    pickle.dump([np.array(vectors),data_labels[chunk*chunk_size:(chunk+1)*chunk_size]], f)
                self.timer.count()
            print("... total time:")
            self.timer.toc()
            
            self.chunks_max = chunks_max
            self.chunk_size = chunk_size
            self.size = len(self.d.data)
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"wb") as f:
                pickle.dump([self.chunks_max, self.chunk_size, self.size], f)
                
            del self.d
            del self.glove
            if hasattr(self,"l"):
                del self.l
        
    #def get(self,i):
    #    label = torch.cuda.LongTensor(self.data_labels[i]).view(1)
    #    x = torch.cuda.FloatTensor(self.data_inputs[i]).unsqueeze(0).unsqueeze(0)
    #
    #    return x, label
    
    def batchify(self,size,num_chunks,shuffle=True):
        """
        Batchifies the dataset.
        
        Args:
            size (int): number of rows in a batch.
            num_chunks (int): number of chunks to preload into memory.
            shuffle (bool): retrieve randomized batches
        """
        self.timer_all = 0 #delme
        
        self.batch_size = size
        # count total number of batches processed
        self.batch_total = 0
        
        self.chunk_current = 0
        self.chunks = np.arange(self.chunks_max)
        self.chunks_step = num_chunks
        if shuffle:
            np.random.shuffle(self.chunks)
        self.shuffle = shuffle
        
        self._next_chunks()
        
    def _next_chunks(self):
        if self.chunk_current < self.chunks_max:
            #if VERBOSE_EPOCHS:
            #    print("Loading next chunks.")
            #    self.timer.tic()
            
            # don't reload the data if there is only one chunk
            if not (hasattr(self,"data_inputs") and self.chunks_max==1):
                self._load_chunks(self.chunks[self.chunk_current:self.chunk_current+self.chunks_step])
            
            # initialize batching
            self.batch_current = 0
            self.batch_max = len(self.data_inputs)/self.batch_size
            self.batches = np.arange(len(self.data_inputs))
            if self.shuffle:
                np.random.shuffle(self.batches)
            
            #if VERBOSE_EPOCHS:
            #    self.timer.toc()
            self.chunk_current += self.chunks_step
            return True
        
        return False
        
    def _load_chunks(self,chunks):
        self._clear_memory()
        
        for c in chunks:
            file = os.path.join(self.filepath,"data."+str(c))
            with open(file,"rb") as f:
                inputs, labels = pickle.load(f)
                try:
                    self.data_inputs = np.concatenate((self.data_inputs,inputs))
                except AttributeError:
                    self.data_inputs = inputs
                try:
                    self.data_labels = np.concatenate((self.data_labels,labels))
                except AttributeError:
                    self.data_labels = labels
            
    def _clear_memory(self):
        if hasattr(self,"data_inputs"):
            del self.data_inputs
        if hasattr(self,"data_labels"):
            del self.data_labels
            
        #self.data_inputs = np.empty((self.chunks_step*self.chunk_size,),dtype="object")
        #self.data_labels = np.empty((self.chunks_step*self.chunk_size,),dtype="int64")
        
    def next_batch(self):
        if not self.batch_current < self.batch_max:
            self._next_chunks()
            
        #print("Getting batch {}/{}".format(self.batch_current,self.batch_max))
        #timer.tic()
        
        # get (shuffled) indices
        i = self.batch_current * self.batch_size
        indices = self.batches[i:(i+self.batch_size)]

        # get labels
        labels = self.data_labels[indices]
        labels = torch.tensor(labels,device=self.device,dtype=torch.long).view(len(labels))#,1)
        
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
        inputs = torch.tensor(inputs,device=self.device,dtype=torch.float).unsqueeze(1)
        
        self.batch_current += 1
        self.batch_total += 1
        
        #timer.toc()
        return inputs, labels
    
    def has_next_batch(self):
        return (self.batch_current < self.batch_max) or (self.chunk_current < self.chunks_max)
        
    def num_classes(self):
        return len(self.classes)