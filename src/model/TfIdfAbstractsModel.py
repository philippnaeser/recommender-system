# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pickle

class TfIdfAbstractsModel(AbstractModel):
    
    persistent_file = os.path.join("..","..","data","processed","abstracts.tfidf.model.pkl")
    
    ##########################################
    def __init__(self,recs=10):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                #,min_df=10
                #,max_df=0.6
        )
        # number of recommendations to return
        self.recs = recs
    
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
        #print(q_v)
        sim = cosine_similarity(q_v,self.stem_matrix)[0]
        o = np.argsort(-sim)
        #print(self.data.chapter_abstract[o][0:10])
        print(self.data.iloc[o[0]].chapter_abstract)
        return [
                list(self.data.iloc[o][0:self.recs].conference_name),
                sim[o][0:self.recs]
                ]
    
    ##########################################
    def query_batch(self,batch):
        """
            Queries the model and returns a list of recommendations for each request.
            
            Args:
                batch[str]: The list of abstracts.
            
            Returns:
                A list of size 'len(batch)' which contains the recommendations for each item of the batch.
                If author not found, the value is None.
                
                str[]: name of the conference
                double[]: confidence scores
        """
        conferences = list()
        confidences = list()
        self.count_init(len(batch))
        
        q_v = (self.stem_vectorizer.transform(batch))
        print("Abstracts transformed.")
        print("Dimensionality of batch: {}".format(q_v.shape))
        sim = cosine_similarity(q_v,self.stem_matrix)
        print("Cosine similarity computed.")
        o = np.argsort(-np.array(sim))
        index = 0
        
        for order in o:
            conferences.append(
                    list(self.data.iloc[order][0:self.recs].conference_name)
            )
            confidences.append(
                    
                    sim[index][order][0:self.recs]
            )
            index += 1
            self.count()
            
        return [conferences,confidences]
    
    ##########################################
    def train(self,data):
        if not self._load_model():
            print("Model not persistent yet. Creating model.")
            #for check in ["abstract","conference","conference_name"]:
            for check in ["chapter_abstract","conference","conference_name"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract.str.decode("unicode_escape"))
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
        with open(TfIdfAbstractsModel.persistent_file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
    
    ##########################################
    def _load_model(self):
        if os.path.isfile(TfIdfAbstractsModel.persistent_file):
            with open(TfIdfAbstractsModel.persistent_file,"rb") as f:
                print("... loading ...")
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))
