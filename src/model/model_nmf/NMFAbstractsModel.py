# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
@author: Andreea
"""

from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemmer from nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import re
import os
import pickle

class NMFAbstractsModel(AbstractModel):
    
    persistent_file_x = os.path.join("..","..","..","data","processed","abstracts.nmf.model.X.pkl")
    persistent_file_lr = os.path.join("..","..","..","data","processed","abstracts.nmf.model.LR.pkl")
    
    ##########################################
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                #,min_df=0.05
                #,max_df=0.8
        )
        # number of recommendations to return
        self.recs = 10
    
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
        transformed_q_v = self.nmf.transform(q_v)
        
        # normalize
        row_sums = transformed_q_v.sum(axis=1)
        row_sums = np.where(row_sums == 0, 0.00000001, row_sums)
        transformed_q_v = transformed_q_v / row_sums[:, np.newaxis]
        
        sim = cosine_similarity(transformed_q_v,self.nmf_L)[0]
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
        
        #print(batch)
        q_v = (self.stem_vectorizer.transform(batch))
        transformed_q_v = self.nmf.transform(q_v)
        print("Abstracts transformed.")
        print(q_v.shape)
        print(transformed_q_v.shape)
        
        # normalize
        row_sums = transformed_q_v.sum(axis=1)
        row_sums = np.where(row_sums == 0, 0.00000001, row_sums)
        transformed_q_v = transformed_q_v / row_sums[:, np.newaxis]
        
        sim = cosine_similarity(transformed_q_v,self.nmf_L)
        print("Cosine similarity computed.")
        o = np.argsort(-np.array(sim))
        
        conference = list()
        confidence = list()
        index = 0
        self.count_init(len(o))
        for order in o:
            conference.append(
                    list(self.data.iloc[order][0:self.recs].conference_name)
            )
            confidence.append(
                    sim[index][order][0:self.recs]
            )
            index += 1
            self.count()
        
            
        return [conference,confidence]
        
    ##########################################
    def train(self,data,topics=100,alpha=2):
        if not self._load_model_x():
            print("Stem matrix not persistent yet. Creating now.")
            #for check in ["abstract","conference","conference_name"]:
            for check in ["chapter_abstract","conference","conference_name"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model_x()
                     
        if not self._load_model_lr():
            print("NMF not persistent yet. Creating now.")
            self.nmf = NMF(
                    n_components = topics
                    ,init = "random"
                    #,beta_loss = "kullback-leibler"
                    ,beta_loss = "frobenius"
                    #,solver = "mu"
                    ,solver = "cd"
                    ,random_state = 0
                    ,verbose = True
                    ,alpha=alpha
            )
            self.nmf_L = self.nmf.fit_transform(self.stem_matrix)
            
            # normalize L
            row_sums = self.nmf_L.sum(axis=1)
            row_sums = np.where(row_sums == 0, 0.00000001, row_sums)
            self.nmf_L = self.nmf_L / row_sums[:, np.newaxis]
            
            #self.nmf_R = self.nmf.components_
            self._save_model_lr()
        
    ##########################################
    def __call__(self, doc):
        # tokenize the input with a regex and stem each token
        return [self.stemmer.stem(t) for t in self.token_pattern.findall(doc)]
    
    ##########################################
    def _get_word_freq(self, matrix, vectorizer):
        '''Function for generating a list of (freq, word)'''
        return sorted([(matrix.getcol(idx).sum(), word) for word, idx in vectorizer.vocabulary_.items()], reverse=True)
    
    ##########################################
    def _save_model_x(self):
        with open(NMFAbstractsModel.persistent_file_x,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
            
    ##########################################
    def _save_model_lr(self):
        with open(NMFAbstractsModel.persistent_file_lr,"wb") as f:
            pickle.dump([self.nmf, self.nmf_L], f)
    
    ##########################################
    def _load_model_x(self):
        if os.path.isfile(NMFAbstractsModel.persistent_file_x):
            print("Loading persistent models: X")
            with open(NMFAbstractsModel.persistent_file_x,"rb") as f:
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_lr(self):
        if os.path.isfile(NMFAbstractsModel.persistent_file_lr):
            print("Loading persistent models: LR")
            with open(NMFAbstractsModel.persistent_file_lr,"rb") as f:
                self.nmf, self.nmf_L = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))
