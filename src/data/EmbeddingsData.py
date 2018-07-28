# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 12:59:31 2018

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
from nltk.tokenize import sent_tokenize
import pandas as pd
import gzip
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#### TRAINING PARAMETERS

#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "small" 

class EmbeddingsData():
    
    path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "data",
                "interim",
                "embeddings_training"
                )

    def __init__(self, data_which = "small"):
        
        self.filepath = os.path.join(self.path_persistent, "train-data-" + data_which)
        self.timer = Timer()
        
        if os.path.isdir(self.filepath):
            print("Training data already transformed.")
            self.sentences = gensim.models.word2vec.LineSentence(os.path.join(self.filepath, "abstracts.gz"))
            
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

            print("Transforming and saving abstracts.")
            self.timer.tic()
            
            self.training_data = self._sentenceToLine()
            file = os.path.join(self.filepath, "abstracts.gz")
            with gzip.open(file, "wb") as f:
                f.writelines(sentence.encode("utf-8") for sentence in self.training_data)

            print("... total time:")
            self.timer.toc() 
            
            del self.d   
            
    def _sentenceToLine(self):
        """
        Splits abstracts in one line = one sentence
        
        Returns:
            text[list]: list with the processed sentences of the abstracts
        """
        text = list()
        count_lines = 0
        
        for abstract in self.training_data:
            line = 0
            for sentence in sent_tokenize(abstract):
                text.append(sentence.rstrip(".") + "\n")
                line += 1
            count_lines += line
            
        print("Finsihed transforming {} abstracts with {} lines.".format(
                len(self.training_data), count_lines))
                
        return text
    
    def getTrainingData(self):
        self.sentences = gensim.models.word2vec.LineSentence(os.path.join(self.filepath, "abstracts.gz"))
        return self.sentences
    
##Example
#embedder = EmbeddingsData(data_which = DATA_TRAIN)
#train_inputs = embedder.getTrainingData()
