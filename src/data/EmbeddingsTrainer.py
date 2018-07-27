# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:45:40 2018

@author: Andreea
"""

#import sys
import os

#sys.path.insert(0, os.path.join(os.getcwd()))
#sys.path.insert(0, os.path.join(os.getcwd(),".."))
#sys.path.insert(0, os.path.join(os.getcwd(),"..","..","model"))
#sys.path.insert(0, os.path.join(os.getcwd(),"..","..", "..", "neuralnets"))

from EmbeddingsData import EmbeddingsData
import time
import logging
from gensim.models import Word2Vec, FastText, KeyedVectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Timer:
    start_time = []
    
    ### start runtime check
    def tic(self):
        self.start_time.append(time.time())
    
    ### print runtime information
    def toc(self):
        diff = (time.time() - self.start_time.pop())
        print("Timer :: toc --- %s seconds ---" % diff)
        return diff
        
    def set_counter(self,c,max=100):
        self.counter_max = c
        self.counter = 0
        self.checkpoint = int(self.counter_max/max)
        self.step = self.checkpoint
        self.tic()
        
    def count(self,add=1):
        self.counter = self.counter + add
        
        if (self.counter >= self.checkpoint):
            print("Timer :: Checkpoint reached: {}%".format(int(self.counter*100/self.counter_max)))
            self.toc()
            self.checkpoint += self.step
            if self.checkpoint <= self.counter_max:
                self.tic()

#### TRAINING PARAMETERS
                
#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "small" 
                
# set a name which is used as directory for the saved models 
# and word vectors in "data/processed/word_embeddings/<name>"
EMBEDDING_NAME = "w2v_50d"

#The model used. One of {"word2vec","fasttext"}
EMBEDDING_MODEL = "word2vec" 

SIZE = 50
WINDOW = 10
MIN_COUNT = 5
WORKERS= 10

#Training algorithm: skip-gram if sg=1, otherwise CBOW.
SG = 0

ITER = 10

class EmbeddingsTrainer():
    
    def __init__(self, data_which, embedding_name):
        
        self.path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            #"..",
            "data",
            "processed",
            "word_embeddings"
        )

        self.filepath = os.path.join(self.path_persistent, embedding_name)
        
        self.paths = {
            "word2vec-model":os.path.join(self.filepath, "word2vec.model"),
            "word2vec-wv":os.path.join(self.filepath, "wordvectors.kv"),
            "fasttext-model":os.path.join(self.filepath, "fastext.model"),
            "fasttext-wv":os.path.join(self.filepath, "fasttext.kv")
            }
        
        embedder = EmbeddingsData(data_which)
        self.sentences = embedder.getTrainingData()
        self.timer = Timer()
            
    def train_model(self, embedding_model="word2vec", 
                    size = 50, window = 10, min_count = 5, 
                    workers = 10, sg = 0, iterations = 10):
        """
        Trains a word embeddings model.
        
        Args:
            model (str): The model used. One of {"word2vec","fasttext"}.
        """
        
        if os.path.isdir(self.filepath):
            print('Loading model.')
            self.model = KeyedVectors.load(self.paths[embedding_model + "-model"])
            
            print('Loading trained word vectors.')
            self.word_vectors = KeyedVectors.load(
                    self.paths[embedding_model + "-wv"], 
                    mmap='r'
                    )
            
        else:
            print("Model not persistent yet.")
            os.mkdir(self.filepath)
            
            print("Starting training the model.")
            self.timer.tic()
            if embedding_model == "word2vec":
                self.model = Word2Vec(
                    self.sentences,
                    size=size,
                    window = window,
                    min_count= min_count,
                    workers= workers,
                    sg = sg,
                    iter = iterations
                    )
            else:
                self.model = FastText(
                        self.sentences,
                    size=size,
                    window = window,
                    min_count= min_count,
                    workers= workers,
                    sg = sg,
                    iter = iterations
                    )
                
            print("... total training time:")
            self.timer.toc()

            print("Saving model to disk.")
            self.model.save(self.paths[embedding_model + "-model"])

            print("Saving word vectors to disk.")
            self.word_vectors = self.model.wv
            del self.model
            self.word_vectors.save(self.paths[embedding_model + "-wv"])
        
# Example:
trainer = EmbeddingsTrainer(
        data_which = DATA_TRAIN,
        embedding_name = EMBEDDING_NAME
        )
trainer.train_model(
        embedding_model = EMBEDDING_MODEL,
        size = SIZE,
        window = WINDOW,
        min_count = MIN_COUNT,
        workers = WORKERS,
        sg = SG,
        iterations = ITER 
        )
