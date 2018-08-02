# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 20:30:27 2018

@author: Andreea
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from EmbeddingsLoader import EmbeddingsLoader
from collections import defaultdict
import numpy as np


class TFIDFEmbeddingsVectorizer():
    
    def __init__(self, embedding_model, pretrained = True):
        
        parser = EmbeddingsLoader()
        
        self.word2vec = parser.load_model(embedding_model, pretrained)
        self.word2weight = None
        self.dim = parser.get_dimension(embedding_model)
        
    def fit(self, X, y):
        
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        
        max_idf = max(tfidf.idf_)
        
        self.word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
                )
        
        return self
        
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                        for w in words if w in self.word2vec] or 
                        [np.zeros(self.dim)], axis=0)
                for words in X
                ])
                