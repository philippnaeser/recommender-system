# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 20:30:27 2018

@author: Steff
@author: Andreea
"""

import os
#import sys
#sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from gensim.models.keyedvectors import KeyedVectors
import spacy
import numpy as np

"""
import time
class Timer:
    start_time = []
    
    ### start runtime check
    def tic(self):
        self.start_time.append(time.time())
    
    ### print runtime information
    def toc(self):
        print("Timer :: toc --- %s seconds ---" % (time.time() - self.start_time.pop()))
        
    def set_counter(self,c,max=100):
        self.counter_max = c
        self.counter = 0
        self.checkpoint = int(self.counter_max/max)
        self.step = self.checkpoint
        self.tic()
        
    def count(self,add=1):
        self.counter = self.counter + add
        
        if (self.counter > self.checkpoint):
            print("Timer :: Checkpoint reached: {}%".format(int(self.counter*100/self.counter_max)))
            self.toc()
            self.tic()
            self.checkpoint += self.step
"""

class EmbeddingsParser:

    path_embeddings = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "external"
    )

    paths = {
            "6d50":os.path.join(path_embeddings,"glove.6B.50d.txt"),
            "6d50-w2v":os.path.join(path_embeddings,"glove.6B.50d-w2v.txt"),
            "6d50-folder":os.path.join(path_embeddings,"glove.6B.50d-spacy",""),
            "6d100":os.path.join(path_embeddings,"glove.6B.100d.txt"),
            "6d100-w2v":os.path.join(path_embeddings,"glove.6B.100d-w2v.txt"),
            "6d100-folder":os.path.join(path_embeddings,"glove.6B.100d-spacy",""),
            "6d200":os.path.join(path_embeddings,"glove.6B.200d.txt"),
            "6d200-w2v":os.path.join(path_embeddings,"glove.6B.200d-w2v.txt"),
            "6d200-folder":os.path.join(path_embeddings,"glove.6B.200d-spacy",""),
            "6d300":os.path.join(path_embeddings,"glove.6B.300d.txt"),
            "6d300-w2v":os.path.join(path_embeddings,"glove.6B.300d-w2v.txt"),
            "6d300-folder":os.path.join(path_embeddings,"glove.6B.300d-spacy",""),
            "42d300":os.path.join(path_embeddings,"glove.42B.300d.txt"),
            "42d300-w2v":os.path.join(path_embeddings,"glove.42B.300d-w2v.txt"),
            "42d300-folder":os.path.join(path_embeddings,"glove.42B.300d-spacy",""),
            "840d300":os.path.join(path_embeddings,"glove.840B.300d.txt"),
            "840d300-w2v":os.path.join(path_embeddings,"glove.840B.300d-w2v.txt"),
            "840d300-folder":os.path.join(path_embeddings,"glove.840B.300d-spacy",""),
            "word2vec":os.path.join(path_embeddings,"GoogleNews-vectors-negative300.bin"),
            "word2vec-w2v":os.path.join(path_embeddings,"word2vec.300d-w2v.txt"),
            "word2vec-folder":os.path.join(path_embeddings,"word2vec-spacy","")
    }
    
    lengths = {
            "6d50":50,
            "6d100":100,
            "6d200":200,
            "6d300":300,
            "42d300":300,
            "840d300":300,
            "word2vec":300
    }
    
    models = {}
    
    nlp = spacy.load("en",vectors=False)
    
    #timer = Timer()
    
    def load_model(self,model):
        """
        Loads a pre-trained word embedding to be used by this parser.
        
        Args:
            model (str): The model used. One of {"6d50","6d100","6d200","6d300","42d300","840d300", "word2vec"}.
        """
        try:
            self.nlp = spacy.load(self.paths[model + "-folder"])
            self.length = self.lengths[model]
            
        except OSError:
            if not os.path.isfile(self.paths[model + "-w2v"]):
                if model == "word2vec":
                    print("Word2Vec format not present, generating it.")
                    self.models[model] = KeyedVectors.load_word2vec_format(self.paths["word2vec"], binary=True)
                    self.models[model].save_word2vec_format(self.paths[model + "-w2v"], binary=False)       
                else:
                    from gensim.scripts.glove2word2vec import glove2word2vec
                    print("Word2Vec format not present, generating it.")
                    glove2word2vec(glove_input_file=self.paths[model], word2vec_output_file=self.paths[model + "-w2v"])
                        
            try:
                self.current_model = self.models[model + "-w2v"]
                self.length = self.lengths[model]
            except KeyError:
                if model == "word2vec":
                    print("Word2Vec not loaded yet, loading it.")
                else:
                    print("Glove not loaded yet, loading it.")
                self.models[model + "-w2v"] = KeyedVectors.load_word2vec_format(self.paths[model + "-w2v"], binary=False)
                self.current_model = self.models[model + "-w2v"]
                self.length = self.lengths[model]
            
            print("Setting up spacy vocab.")
            
            count = 0
            for word, o in self.current_model.vocab.items():
                count += 1
                self.nlp.vocab.set_vector(word, self.current_model[word])
            
            print("Done ({}). Saving to disk.".format(count))
            
            try:
                self.nlp.to_disk(self.paths[model + "-folder"])
            except FileNotFoundError:
                os.mkdir(self.paths[model + "-folder"])
                self.nlp.to_disk(self.paths[model + "-folder"])
    
    #################################################    
    def transform_matrix(self,sentence):
        """
        Transform a string into a matrix containing word embeddings as rows.
        
        Args:
            sentence (str): The string to be transformed. Can be a sentence or a whole document.
            
        Returns:
            A numpy matrix that contains word embeddings as rows of each word given by "sentence".
        """
        for w in self.nlp(sentence):
            try:
                m = np.concatenate((m,[w.vector]),axis=0)
            except NameError:
                m = np.array([w.vector])
            
        return m
    
    #################################################
    def transform_vector(self,sentence):
        """
        Transform a string into a vector containing concatenated word embeddings.
        
        Args:
            sentence (str): The string to be transformed. Can be a sentence or a whole document.
            
        Returns:
            A numpy array that contains concatenated word embeddings for each word given by "sentence".
        """
        
        for w in self.nlp(sentence):
            try:
                m = np.append(m,w.vector)
            except NameError:
                m = np.array(w.vector)
            
        return m
    
    #################################################
    def transform_vectors(self,sentences,batch_size=100):
        """
        Transform a list of strings into a list of vectors containing concatenated word embeddings.
        
        Args:
            sentence list(str): The list of strings to be transformed.
            
        Returns:
            A list of numpy arrays that contain concatenated word embeddings for each word given by "sentence".
        """
        
        vectors = list()
        for doc in self.nlp.tokenizer.pipe(sentences, batch_size=batch_size):
            m = np.empty(len(doc)*self.length)
            for i, w in enumerate(doc):
                m[(i*self.length):((i+1)*self.length)] = w.vector
            vectors.append(m)
            
        return vectors
    
    #################################################
    def transform_avg_vectors(self,sentences,batch_size=100):
        """
        Transform a list of strings into a list of vectors containing averaged word embeddings.
        
        Args:
            sentence list(str): The list of strings to be transformed.
            
        Returns:
            A list of numpy arrays that contain averaged word embeddings for each word given by "sentence".
        """
        
        vectors = list()
        for doc in self.nlp.tokenizer.pipe(sentences, batch_size=batch_size):
            m = np.empty(self.length)
            for i, w in enumerate(doc):
                m = np.add(m,w.vector)
            vectors.append(m/len(doc))
            
        return vectors
    
    
        
## Example:
#parser = EmbeddingsParser()
#parser.load_model("6d50")
#test = parser.transform_vector("oi mate, what's going on?")
#print(test.shape)