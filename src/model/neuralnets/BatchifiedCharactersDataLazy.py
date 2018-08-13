# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:42:14 2018

@author: Steff
"""
import os
import pickle
import sys
import string
import unicodedata

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from DataLoader import DataLoader as SciGraphLoader
from TimerCounter import Timer

class BatchifiedCharactersDataLazy():
    
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
    def __init__(self,use_cuda=True,training=True,data_which="small",classes=None,max_document_size=1280):
        if use_cuda:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
            
        if training:
            data_type = "training"
        else:
            data_type = "test"
            
        filepath = os.path.join(self.path_persistent,data_type+"-data-characters-lazy-"+data_which)
        
        self.letters = string.ascii_letters[0:26]                #alpha
        self.letters += "".join(str(i) for i in range(10))       #numeric
        self.letters += "-,;.!?:\"'“”/\\|_@#$%ˆ&*˜+-=<>()[]{} "  #special [Zhang, 2016]
        # add additional None character
        self.letters_size = len(self.letters)+1
        self.letters_dict = {}
        for i,l in enumerate(self.letters,1):
            self.letters_dict[l] = i
        # add additional None character
        self.letters_dict[None] = 0
            
        self.max_document_size = max_document_size
        
        if os.path.isdir(filepath):
            print("Loading {} data from disk.".format(data_type))
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
                d.training_data_for_abstracts(data_which)
            else:
                d.test_data_for_abstracts(data_which)
            
            print("Normalizing abstracts.")
            self.data_abstracts = d.data[["chapter_abstract"]].copy().chapter_abstract.str.lower()
            self.data_abstracts = self.data_abstracts.apply(lambda x: self.normalize_string(str(x)))
            
            self.data_labels = d.data["conferenceseries"]
            del d
            
            # initialize labels
            if training:
                print("Initializing labels.")
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
                print("Updating labels.")
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
                
            print("Saving normalized abstracts.")
            file = os.path.join(filepath,"data_abstracts.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.data_abstracts, f)
            
        self.data_size = len(self.data_labels)
        
    ####################################################################
    def normalize_string(self,s):
        return "".join(
            c for c in unicodedata.normalize("NFD",s)
            if unicodedata.category(c) != "Mn" and c in self.letters
        )

    ####################################################################
    def letter_to_index(self,l):
        return self.letters_dict[l]
    
    ####################################################################    
    def normalized_text_to_tensor(self,text):
        tensor = np.zeros(
                (1,self.letters_size,self.max_document_size),
                dtype=np.float32
        )
        for i, c in enumerate(text[0:self.max_document_size]):
            tensor[0][self.letter_to_index(c)][i] = 1
        #return torch.from_numpy(tensor)
        return torch.tensor(tensor,device=self.device)
    
    ####################################################################    
    def normalized_batch_to_tensor(self,batch):
        tensor = np.zeros(
                (len(batch),self.letters_size,self.max_document_size),
                dtype=np.float32
        )
        for b_i, text in enumerate(batch):
            for c_i, c in enumerate(text[0:self.max_document_size]):
                tensor[b_i][self.letter_to_index(c)][c_i] = 1
        #return torch.from_numpy(tensor)
        return torch.tensor(tensor,device=self.device)
    
    ####################################################################    
    def unnormalized_text_to_tensor(self,text):
        text = self.normalize_string(text[0:self.max_document_size])
        
        tensor = np.zeros(
                (1,self.letters_size,self.max_document_size),
                dtype=np.float32
        )
        for i, c in enumerate(text):
            tensor[0][self.letter_to_index(c)][i] = 1
        #return torch.from_numpy(tensor)
        return torch.tensor(tensor,device=self.device)
    
    ####################################################################    
    def unnormalized_batch_to_tensor(self,batch):
        tensor = np.zeros(
                (len(batch),self.letters_size,self.max_document_size),
                dtype=np.float32
        )
        for b_i, text in enumerate(batch):
            text = self.normalize_string(text[0:self.max_document_size])
            for c_i, c in enumerate(text[0:self.max_document_size]):
                tensor[b_i][self.letter_to_index(c)][c_i] = 1
        #return torch.from_numpy(tensor)
        return torch.tensor(tensor,device=self.device)
    
    ####################################################################
    def batchify(self,batch_size,shuffle=True):
        self.batch_size = batch_size
        self.batches = np.arange(self.data_size,step=batch_size)
        
        self.batch_current = 0
        self.batch_max = len(self.batches)
        
        if shuffle:
            order = np.arange(len(self.data_abstracts))
            np.random.shuffle(order)
            self.data_abstracts = self.data_abstracts[order]
            self.data_labels = self.data_labels[order]
        
    ####################################################################
    def has_next_batch(self):
        return self.batch_current < self.batch_max
        
    ####################################################################
    def next_batch(self):
        i_from = self.batch_current*self.batch_size
        i_to = i_from + self.batch_size
        
        batch = self.normalized_batch_to_tensor(self.data_abstracts[
                i_from:i_to
        ])
    
        labels = self.data_labels[i_from:i_to]
        labels = torch.tensor(
                labels,
                device=self.device,
                dtype=torch.long
        ).view(len(labels))
    
        self.batch_current += 1
        return batch, labels

    ####################################################################
    def num_classes(self):
        return len(self.classes)
    
    ####################################################################
    def num_letters(self):
        return self.letters_size


#data = BatchifiedCharactersDataLazy(use_cuda=False)        
#timer = Timer()

#print("Starting test")
#timer.tic()
#data.batchify(1000,shuffle=True)
#while data.has_next_batch():
#    test = data.next_batch()
#timer.toc()



"""
def test():
    for i,d in enumerate(data.data_abstracts[0:10000]):
        data._textToTensor(d)

if __name__ == "__main__":
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test",number=1))
"""