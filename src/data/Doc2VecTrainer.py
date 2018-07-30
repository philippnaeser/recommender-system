# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:32:08 2018

@author: Andreea
"""

import os
from Doc2VecData import Doc2VecData
from gensim.models.doc2vec import Doc2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#### TRAINING PARAMETERS
                
#The size of the training data. One of {"small", "medium", "all"}
DATA_TRAIN = "all" 
                
# set a name which is used as directory for the saved models 
# and word vectors in "data/processed/word_embeddings/<name>"
EMBEDDING_NAME = "d2v_100d_w5_HS" 

SIZE = 100
WINDOW = 5
ALPHA = 0.025
MIN_ALPHA = 0.025
MIN_COUNT = 2
WORKERS= 20
ITER = 20
HS = 1

class Doc2VecTrainer():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "doc_embeddings"
        )
    
    def __init__(self, embedding_name, data_which = "all"):
        
        self.filepath = os.path.join(self.path_persistent, embedding_name)
        
        parser = Doc2VecData(data_which)
        self.docs = parser.getTrainingData()
            
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
        
        if os.path.isfile(self.filepath):
            print('Loading model.')
            self.model = Doc2Vec.load(self.filepath)
            
        else:
            print("Model not persistent yet.")            
            print("Starting training the model.")
            
            self.model = Doc2Vec(
                    dm = 1,
                    size = size,
                    window = window,
                    alpha = alpha,
                    min_alpha = min_alpha,
                    min_count = min_count,
                    workers = workers,
                    hs = hs                   
                    )  
            
            self.model.build_vocab(self.docs)

#            self.model.train(
#                        self.docs, 
#                        total_examples = self.model.corpus_count,
#                        epochs = iterations
#                        )
            for epoch in range(iterations):
                print('Iteration {}\n'.format(epoch+1))
                self.model.train(
                        self.docs, 
                        total_examples = self.model.corpus_count,
                        epochs = self.model.iter
                        )
                self.model.alpha -= 0.002    #decrease the learning rate
                self.model.min_alpha = self.model.alpha #fix the learning rate, no decay

            print("Saving model to disk.")
            self.model.save(self.filepath)           
            
            
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
    