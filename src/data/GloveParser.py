# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 20:30:27 2018

@author: Steff
"""

import os
#import sys
#sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from gensim.models.keyedvectors import KeyedVectors
import spacy
import numpy as np

class GloveParser:

    path_glove = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "external"
    )

    paths = {
            "6d50":os.path.join(path_glove,"glove.6B.50d.txt"),
            "6d50-w2v":os.path.join(path_glove,"glove.6B.50d-w2v.txt"),
            "6d50-folder":os.path.join(path_glove,"glove.6B.50d-spacy",""),
            "6d100":os.path.join(path_glove,"glove.6B.100d.txt"),
            "6d100-w2v":os.path.join(path_glove,"glove.6B.100d-w2v.txt"),
            "6d100-folder":os.path.join(path_glove,"glove.6B.100d-spacy",""),
            "6d200":os.path.join(path_glove,"glove.6B.200d.txt"),
            "6d200-w2v":os.path.join(path_glove,"glove.6B.200d-w2v.txt"),
            "6d200-folder":os.path.join(path_glove,"glove.6B.200d-spacy",""),
            "6d300":os.path.join(path_glove,"glove.6B.300d.txt"),
            "6d300-w2v":os.path.join(path_glove,"glove.6B.300d-w2v.txt"),
            "6d300-folder":os.path.join(path_glove,"glove.6B.300d-spacy",""),
            "42d300":os.path.join(path_glove,"glove.42B.300d.txt"),
            "42d300-w2v":os.path.join(path_glove,"glove.42B.300d-w2v.txt"),
            "42d300-folder":os.path.join(path_glove,"glove.42B.300d-spacy",""),
            "840d300":os.path.join(path_glove,"glove.840B.300d.txt"),
            "840d300-w2v":os.path.join(path_glove,"glove.840B.300d-w2v.txt"),
            "840d300-folder":os.path.join(path_glove,"glove.840B.300d-spacy","")
    }
    
    models = {}
    
    nlp = spacy.load("en",vectors=False)
    
    def load_model(self,model):
        """
        Loads a pre-trained word embedding to be used by this parser.
        
        Args:
            model (str): The model used. One of {"6d50","6d100","6d200","6d300","42d300","840d300"}.
        """
        try:
            self.nlp = spacy.load(self.paths[model + "-folder"])
            
        except OSError:
            if not os.path.isfile(self.paths[model + "-w2v"]):
                from gensim.scripts.glove2word2vec import glove2word2vec
                print("Word2Vec format not present, generating it.")
                glove2word2vec(glove_input_file=self.paths[model], word2vec_output_file=self.paths[model + "-w2v"])
            
            #model += "-w2v"
            
            try:
                self.current_model = self.models[model + "-w2v"]
            except KeyError:
                print("Glove not loaded yet, loading it.")
                self.models[model + "-w2v"] = KeyedVectors.load_word2vec_format(self.paths[model + "-w2v"], binary=False)
                self.current_model = self.models[model + "-w2v"]
            
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
        
    def transform(self,sentence):
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
        
## Example:
#parser = GloveParser()
#parser.load_model("840d300")
#test = parser.transform("oi mate, what's going on?")
#print(test.shape)