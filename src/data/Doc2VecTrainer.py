# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:32:08 2018

@author: Andreea
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "model"))
sys.path.insert(0, os.path.join(os.getcwd(),"..", "..", "neuralnets"))

from Doc2VecData import Doc2VecData
from TimerCounter import Timer
from gensim.models.doc2vec import Doc2Vec
import random
import logging
from gensim.models.keyedvectors import Doc2VecKeyedVectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#### TRAINING PARAMETERS
                
#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "all" 
                
# set a name which is used as directory for the saved models 
# and word vectors in "data/processed/word_embeddings/<name>"
EMBEDDING_NAME = "d2v_nsf_100d_w5_HS" 

SIZE = 100
WINDOW = 5
ALPHA = 0.025
MIN_ALPHA = 0.025
MIN_COUNT = 2
WORKERS= 20
ITER = 20
HS = 0

class Doc2VecTrainer():
    
    def __init__(self, embedding_name, data_which = "all"):
        
        self.path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "doc_embeddings"
        )

        self.paths = {
            "model":os.path.join(self.path_persistent, embedding_name + ".model"),
            "vectors":os.path.join(self.path_persistent, embedding_name + ".bin")
            }
        
        parser = Doc2VecData(data_which)
        self.docs = parser.getTrainingData()
        self.timer = Timer()
            
    def train_model(self, size = 100, window = 5, alpha = 0.025, 
                    min_alpha = 0.025, min_count = 2, workers = 20, 
                    iterations = 20, hs = 0):
        """
        Trains a doc2vec model.
        
        Args:
            size (int): Dimensionality of the word vectors.
            window (int):  Maximum distance between the current and predicted word within a sentence.
            alpha (float): The initial learning rate.
            min_alpha (float): Learning rate will linearly drop to min_alpha as training progresses.
            min_count (int): Ignores all words with lower frequency than this.
            workers (int): Number of worker threads to train the model.
            iterations (int): Number of iterations (epochs) over the corpus.
            hs (int): If 1, hierarchical softmax will be used for model training. 
                If 0, and negative is non-zero, negative sampling will be used.
        """
        
        if os.path.isfile(self.paths["model"]):
            print('Loading model.')
            self.model = Doc2Vec.load(self.paths["model"])
            
        else:
            print("Model not persistent yet.")            
            print("Starting training the model.")
            self.timer.tic()
            
            model = Doc2Vec(
                    dm = 1,
                    size = size,
                    window = window,
                    alpha = alpha,
                    min_alpha = min_alpha,
                    min_count = min_count,
                    workers = workers,
                    hs = hs                   
                    )  
            
            model.build_vocab(self.docs)
            
            for epoch in range(iterations):
                print('Iteration {}\n'.format(epoch))
                #random.shuffle(self.docs)
                model.train(self.docs)
                model.alpha -= 0.002    #decrease the learning rate
                model.min_alpha = model.alpha #fix the learning rate, no decay

            print("... total training time:")
            self.timer.toc()
            
            print("Saving model to disk.")
            self.model.save(self.paths["model"])           
            

        if os.path.isfile(self.paths["vectors"]):
            print('Loading trained word vectors.')
            self.vectors = Doc2VecKeyedVectors.load_word2vec_format(
                    self.paths["vectors"], 
                    binary = True
                    )
            
        else:
            print("Saving word vectors to disk.")
            self.vectors = self.model.wv
            
            self.vectors.save_word2vec_format(
                    self.paths["vectors"], 
                    binary = True
                    )
            
            self.model.delete_temporary_training_data(
                    keep_doctags_vectors=True, 
                    keep_inference=True
                    )
         
# Example:
trainer = Doc2VecTrainer(
        embedding_name = EMBEDDING_NAME,
        data_which = DATA_TRAIN
        )
trainer.train_model(
        size = SIZE,
        window = WINDOW,
        alpha = ALPHA,
        min_alpha = MIN_ALPHA,
        min_count = MIN_COUNT,
        workers = WORKERS,
        iterations = ITER,
        hs = HS,
        )
    