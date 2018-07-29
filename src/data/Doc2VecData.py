# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:31:23 2018

@author: Andreea
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..", "model"))
sys.path.insert(0, os.path.join(os.getcwd(),"..", "..", "neuralnets"))

from DataLoader import DataLoader as SciGraphLoader
from TimerCounter import Timer
from gensim.models.doc2vec import TaggedLineDocument
import pandas as pd
import gzip
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#### TRAINING PARAMETERS

#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "small" 

class Doc2VecData():
    
    path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "data",
                "interim",
                "doc2vec_training"
                )

        
    def __init__(self, data_which="small"):
        
        self.filepath = os.path.join(self.path_persistent, "train-data-" + data_which)
        self.timer = Timer()
        
        if os.path.isdir(self.filepath):
            print("Training data already transformed.")
            self.docs = TaggedLineDocument(os.path.join(self.filepath, "abstracts.gz"))
            
        else:
            print("Training data not on disk.")
            os.mkdir(self.filepath)
            print("Loading and preprocessing training data.")
            
            self.timer.tic()
            print("Loading SciGraph.")
            self.d = SciGraphLoader()    
            self.d.training_data(data_which).abstracts()
        
            self.timer.toc()
            
            # drop empty abstracts
            self.d.data.drop(
                list(self.d.data[pd.isnull(self.d.data.chapter_abstract)].index),
                inplace=True
            )
            self.d.data.reset_index(inplace=True)
                    
            
            print("Preprocessing abstracts.")
            self.timer.tic()
            self.training_data = list(
                    self.d.data["chapter_abstract"].str.lower().str.decode("unicode_escape"))
            self.timer.toc()

            self.training_data = self._abstractToLine()
            print("Transforming and saving abstracts.")
            self.timer.tic()
            
            file = os.path.join(self.filepath, "abstracts.gz")
            with gzip.open(file, "wb") as f:
                f.writelines(abstract.encode("utf-8") for abstract in self.training_data)

            print("... total time:")
            self.timer.toc() 
            
            del self.d   
            
    def _abstractToLine(self):
        """
        Splits abstracts in one line = one abstract
        
        Returns:
            text[list]: list with the processed abstracts
        """
        text = list()
        for abstract in self.training_data:
            text.append(abstract + "\n")
        print("Finished transforming {} abstracts.".format(len(self.training_data)))
        
        return text   
    
    def getTrainingData(self):
        self.docs = TaggedLineDocument(os.path.join(self.filepath, "abstracts.gz"))
        return self.docs
    
#Example
parser = Doc2VecData(data_which = DATA_TRAIN)
train_inputs = parser.getTrainingData()