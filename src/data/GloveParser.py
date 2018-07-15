# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 20:30:27 2018

@author: Steff
"""

import os
#import sys
#sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from gensim.models.keyedvectors import KeyedVectors

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
            "6d100":os.path.join(path_glove,"glove.6B.100d.txt"),
            "6d100-w2v":os.path.join(path_glove,"glove.6B.100d-w2v.txt"),
            "6d200":os.path.join(path_glove,"glove.6B.200d.txt"),
            "6d200-w2v":os.path.join(path_glove,"glove.6B.200d-w2v.txt"),
            "6d300":os.path.join(path_glove,"glove.6B.300d.txt"),
            "6d300-w2v":os.path.join(path_glove,"glove.6B.300d-w2v.txt"),
            "42d300":os.path.join(path_glove,"glove.42B.300d.txt"),
            "42d300-w2v":os.path.join(path_glove,"glove.42B.300d-w2v.txt"),
            "840d300":os.path.join(path_glove,"glove.840B.300d.txt"),
            "840d300-w2v":os.path.join(path_glove,"glove.840B.300d-w2v.txt")
    }
    
    models = {}
    
    def load_file(self,model):
        if not os.path.isfile(self.paths[model + "-w2v"]):
            from gensim.scripts.glove2word2vec import glove2word2vec
            print("Word2Vec format not present, generating it.")
            glove2word2vec(glove_input_file=self.paths[model], word2vec_output_file=self.paths[model + "-w2v"])
        
        model += "-w2v"
        
        try:
            return self.models[model]
        except KeyError:
            print("Model not loaded yet, loading it.")
            self.models[model] = KeyedVectors.load_word2vec_format(self.paths[model], binary=False)
            return self.models[model]
        
## Example:
#parser = GloveParser()
#glove = parser.load_file("6d50")