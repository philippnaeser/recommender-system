# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemmer from nltk
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pickle

class AbstractsModel(AbstractModel):
    
    persistent_file = "..\\..\\data\\processed\\abstracts.model.pkl"
    
    ##########################################
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,min_df=0.2
                ,max_df=0.3
        )
    
    ##########################################
    def query_single(self,abstract):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            abstract (str): The abstract as a string.
        
        Returns:
            str[]: name of the conference
            double[]: confidence scores
        """
        q_v = (self.stem_vectorizer.transform([abstract]))
        sim = cosine_similarity(q_v,self.stem_matrix)[0]
        return self.data[np.argsort(
                sim
        )]
    
    ##########################################
    def query_batch(self,batch):
        pass
    
    ##########################################
    def train(self,data):
        if not self._load_model():
            print("Model not persistent yet. Creating model.")
            #for check in ["abstract","conference","conference_name"]:
            for check in ["chapter_abstract","conference","conference_name"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model()
            #print(self.stem_matrix)
        
    ##########################################
    def __call__(self, doc):
        # tokenize the input with a regex and stem each token
        return [self.stemmer.stem(t) for t in self.token_pattern.findall(doc)]
    
    ##########################################
    def _get_word_freq(self, matrix, vectorizer):
        '''Function for generating a list of (freq, word)'''
        return sorted([(matrix.getcol(idx).sum(), word) for word, idx in vectorizer.vocabulary_.items()], reverse=True)
    
    ##########################################
    def _save_model(self):
        with open(AbstractsModel.persistent_file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
    
    ##########################################
    def _load_model(self):
        if os.path.isfile(AbstractsModel.persistent_file):
            with open(AbstractsModel.persistent_file,"rb") as f:
                print("... loading ...")
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))