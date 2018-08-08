# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
@author: Andreea
"""

from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemNMFAbstractsModelmer from nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import re
import os
import pickle

class NMFMaxAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,topics,beta_loss,solver,alpha,random_state=0,verbose=True,init="random",max_iter=200,min_df=0,max_df=1.0,recs=10):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,min_df=min_df
                ,max_df=max_df
        )
        # number of recommendations to return
        self.recs = recs
        
        self.topics = topics
        self.beta_loss = beta_loss
        self.solver = solver
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.alpha=alpha
        self.max_iter=max_iter
        
        description_stem_matrix = "-".join([
                str(min_df),
                str(max_df),
                "{}"
        ])
    
        description_nmf = "-".join([
                str(self.topics),
                str(self.beta_loss),
                self.solver,
                str(self.alpha),
                self.init,
                str(self.random_state),
                str(self.verbose),
                str(self.max_iter),
                "{}"
        ])
    
        self.path = os.path.join("..","..","..","data","processed","model_nmf_max")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file_x = os.path.join(self.path,
                                              "abstracts.nmf.model."+description_stem_matrix+".X.pkl")
        self.persistent_file_lr = os.path.join(self.path,
                                               "abstracts.nmf.model."+description_nmf+".LR.pkl")
    
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
        return self.query_batch([abstract])
            
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
        q_v = (self.stem_vectorizer.transform(batch))
        transformed_q_v = self.nmf.transform(q_v)
        
        # normalize
        row_sums = transformed_q_v.sum(axis=1)
        row_sums = np.where(row_sums == 0, 0.00000001, row_sums)
        transformed_q_v = transformed_q_v / row_sums[:, np.newaxis]
        
        sim = cosine_similarity(transformed_q_v,self.nmf_L)
        o = np.argsort(-np.array(sim))
        
        conferences = list()
        confidences = list()
            
        for index, order in enumerate(o):
            data_conf = np.array(self.data["conferenceseries"])[order]
            data_sim = np.array(sim[index])[order]
            
            conference = list()
            confidence = list()
            i = 0
            while len(conference) < self.recs:
                c = data_conf[i]
                if c not in conference:
                    conference.append(c)
                    confidence.append(data_sim[i])   
                i += 1
                
            conferences.append(
                    conference
            )
            confidences.append(
                    confidence
            )
            
        return [conference,confidence]
        
    ##########################################
    def train(self,data,data_name):
        if not self._load_model_x(data_name):
            print("Stem matrix not persistent yet. Creating now.")
            #for check in ["abstract","conference","conference_name"]:
            for check in ["chapter_abstract","conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model_x(data_name)
        else:
            if len(self.data) != len(data):
                raise ValueError("Mismatch vs. persistent training data size: Loaded: {} <-> Given: {}".format(len(self.data),len(data)))
                     
        if not self._load_model_lr(data_name):
            print("NMF not persistent yet. Creating now.")
            self.nmf = NMF(
                    n_components = self.topics
                    ,init = self.init
                    #,beta_loss = "kullback-leibler"
                    #,beta_loss = "frobenius"
                    ,beta_loss = self.beta_loss
                    #,solver = "mu"
                    #,solver = "cd"
                    ,solver=self.solver
                    ,random_state = self.random_state
                    ,verbose = self.verbose
                    ,alpha=self.alpha
                    ,max_iter=self.max_iter
            )
            self.nmf_L = self.nmf.fit_transform(self.stem_matrix)
            
            # normalize L
            row_sums = self.nmf_L.sum(axis=1)
            row_sums = np.where(row_sums == 0, 0.00000001, row_sums)
            self.nmf_L = self.nmf_L / row_sums[:, np.newaxis]
            
            #self.nmf_R = self.nmf.components_
            self._save_model_lr(data_name)
        
    ##########################################
    def __call__(self, doc):
        # tokenize the input with a regex and stem each token
        return [self.stemmer.stem(t) for t in self.token_pattern.findall(doc)]
    
    ##########################################
    def _get_word_freq(self, matrix, vectorizer):
        '''Function for generating a list of (freq, word)'''
        return sorted([(matrix.getcol(idx).sum(), word) for word, idx in vectorizer.vocabulary_.items()], reverse=True)
    
    ##########################################
    def _file_x(self,data_name):
        return self.persistent_file_x.format(data_name)
    
    ##########################################
    def _file_lr(self,data_name):
        return self.persistent_file_lr.format(data_name)
    
    ##########################################
    def _save_model_x(self,data_name):
        file = self._file_x(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
            
    ##########################################
    def _save_model_lr(self,data_name):
        file = self._file_lr(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.nmf, self.nmf_L], f)
    
    ##########################################
    def _load_model_x(self,data_name):
        file = self._file_x(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: X")
            with open(file,"rb") as f:
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_lr(self,data_name):
        file = self._file_lr(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: LR")
            with open(file,"rb") as f:
                self.nmf, self.nmf_L = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model(self,data_name):
        return self._load_model_x(data_name) & self._load_model_lr(data_name)
    
    ##########################################
    def _has_persistent_model(self,data_name):
        if os.path.isfile(self._file_x(data_name)) and os.path.isfile(self._file_lr(data_name)):
            return True
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))
