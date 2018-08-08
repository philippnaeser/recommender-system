# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:10:43 2018

@author: Andreea
@author: Steff
"""

from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemmer from nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
import os
import pickle

class LSAAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,topics, random_state=0,min_df=0,max_df=1.0,recs=10):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,strip_accents = "unicode"
                ,min_df=min_df
                ,max_df=max_df
        )
        # number of recommendations to return
        self.recs = recs
        
        self.topics = topics
        self.random_state = random_state
        
        description_stem_matrix = "-".join([
                str(min_df),
                str(max_df),
                "{}"
        ])
    
        description_lsa = "-".join([
                str(self.topics),
                str(self.random_state),
                "{}"
        ])
    
        self.path = os.path.join("..","..","..","data","processed","model_lsa")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file_x = os.path.join(self.path,
                                              "abstracts.lsa.model."+description_stem_matrix+".X.pkl")
        self.persistent_file_factors = os.path.join(self.path,
                                               "abstracts.lsa.model."+description_lsa+".Factors.pkl")
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
        transformed_q_v = self.trsvd.transform(q_v)
        print("Abstracts transformed.")
        print(q_v.shape)
        print(transformed_q_v.shape)
        
        sim = cosine_similarity(transformed_q_v,self.transformed_matrix)
        print("Cosine similarity computed.")
        o = np.argsort(-np.array(sim))
        
        conference = list()
        confidence = list()
        index = 0
        #self.count_init(len(o))
        for order in o:
            conference.append(
                    list(self.data.iloc[order][0:self.recs].conferenceseries)
            )
            confidence.append(
                    list(sim[index][order][0:self.recs])
            )
            index += 1
            #self.count()
            
        return [conference,confidence]
    
   ##########################################
    def train(self, data, data_name):
        if not self._load_model_x(data_name):
            print("Stem matrix not persistent yet. Creating now.")
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model_x(data_name)
        else:
           if len(self.data) != len(data):
                raise ValueError("Mismatch vs. persistent training data size: Loaded: {} <-> Given: {}".format(len(self.data),len(data)))
        
        if not self._load_model_factors(data_name):
            print("SVD not persistent yet. Creating now.")
            self.trsvd = TruncatedSVD(
                    n_components=self.topics
                    , random_state=self.random_state
                    )
            self.transformed_matrix = self.trsvd.fit_transform(self.stem_matrix)
            self._save_model_factors(data_name)
       
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
    def _file_factors(self,data_name):
        return self.persistent_file_factors.format(data_name)
    
    ##########################################
    def _save_model_x(self,data_name):
        file = self._file_x(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
            
    ##########################################
    def _save_model_factors(self,data_name):
        file = self._file_factors(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.trsvd, self.transformed_matrix], f)
    
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
    def _load_model_factors(self,data_name):
        file = self._file_factors(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: Factors")
            with open(file,"rb") as f:
                self.trsvd, self.transformed_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model(self,data_name):
        return self._load_model_x(data_name) & self._load_model_factors(data_name)
    
    ##########################################
    def _has_persistent_model(self,data_name):
        if os.path.isfile(self._file_x(data_name)) and os.path.isfile(self._file_factors(data_name)):
            return True
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))