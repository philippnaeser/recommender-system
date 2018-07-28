# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:45:40 2018

@author: Andreea
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..", "model"))
sys.path.insert(0, os.path.join(os.getcwd(),"..", "..", "neuralnets"))

from EmbeddingsData import EmbeddingsData
from TimerCounter import Timer
import logging
from gensim.models import Word2Vec, FastText, KeyedVectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#### TRAINING PARAMETERS
                
#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "all" 
                
# set a name which is used as directory for the saved models 
# and word vectors in "data/processed/word_embeddings/<name>"
EMBEDDING_NAME = "w2v_50d_w5_CBOW_HS" 

#The model used. One of {"word2vec","fasttext"}
EMBEDDING_MODEL = "word2vec"
SIZE = 50
WINDOW = 5
MIN_COUNT = 2
WORKERS= 20

#Architecture: skip-gram if sg=1, otherwise CBOW.
SG = 0

#Training algorithm: If 1, hierarchical softmax will be used for model training. 
#If 0, and negative is non-zero, negative sampling will be used
HS = 1

ITER = 1

class EmbeddingsTrainer():
    
    def __init__(self, embedding_name, data_which = "all"):
        
        self.path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "word_embeddings"
        )

        self.paths = {
            "model":os.path.join(self.path_persistent, embedding_name + ".model"),
            "wordvectors":os.path.join(self.path_persistent, embedding_name + ".bin")
            }
        
        embedder = EmbeddingsData(data_which)
        self.sentences = embedder.getTrainingData()
        self.timer = Timer()
            
    def train_model(self, embedding_model="word2vec", 
                    size = 50, window = 10, min_count = 5, 
                    workers = 10, sg = 0, hs = 1, iterations = 10):
        """
        Trains a word embeddings model.
        
        Args:
            model (str): The model used. One of {"word2vec","fasttext"}.
        """
        
        if os.path.isfile(self.paths["model"]):
            print('Loading model.')
            self.model = KeyedVectors.load(self.paths["model"])
            
        else:
            print("Model not persistent yet.")            
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
                    hs = hs,
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
                    hs = hs,
                    iter = iterations,
                    word_ngrams = 1,
                    min_n = 3,
                    max_n = 6
                    )
                
            print("... total training time:")
            self.timer.toc()
            
            print("Saving model to disk.")
            self.model.save(self.paths["model"])

        if os.path.isfile(self.paths["wordvectors"]):
            print('Loading trained word vectors.')
            self.word_vectors = KeyedVectors.load_word2vec_format(
                    self.paths["wordvectors"], 
                    binary = True
                    )
            
        else:
            print("Saving word vectors to disk.")
            self.word_vectors = self.model.wv
            del self.model
            self.word_vectors.save_word2vec_format(
                    self.paths["wordvectors"], 
                    binary = True
                    )
        
# Example:
trainer = EmbeddingsTrainer(
        embedding_name = EMBEDDING_NAME,
        data_which = DATA_TRAIN
        )
trainer.train_model(
        embedding_model = EMBEDDING_MODEL,
        size = SIZE,
        window = WINDOW,
        min_count = MIN_COUNT,
        workers = WORKERS,
        sg = SG,
        hs = HS,
        iterations = ITER 
        )
