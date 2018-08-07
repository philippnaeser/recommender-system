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

class TfIdfMaxAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,recs=10,min_df=0,max_df=1.0):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                #,strip_accents = "unicode"
                ,min_df=min_df
                ,max_df=max_df
        )
        # number of recommendations to return
        self.recs = recs
        
        description = "-".join([
                str(min_df),
                str(max_df),
                "{}"
        ])
        
        self.path = os.path.join("..","..","..","data","processed","model_tfidf_max")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file = os.path.join(
                self.path,
                "abstracts.tfidf.model-"+description+".pkl"
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
        conferences = list()
        confidences = list()
        #self.count_init(len(batch))
        
        q_v = (self.stem_vectorizer.transform(batch))
        #print("Abstracts transformed.")
        #print("Dimensionality of batch: {}".format(q_v.shape))
        sim = cosine_similarity(q_v,self.stem_matrix)
        #print("Cosine similarity computed.")
        o = np.argsort(-np.array(sim))

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
            #self.count()
            
        return [conferences,confidences]
    
    ##########################################
    def train(self,data,data_name):
        if not self._load_model(data_name):
            print("Model not persistent yet. Creating model.")
            #for check in ["abstract","conference","conference_name"]:
            for check in ["chapter_abstract","conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model(data_name)
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
    def _file(self,data_name):
        return self.persistent_file.format(data_name)
    
    ##########################################
    def _save_model(self,data_name):
        file = self._file(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
    
    ##########################################
    def _load_model(self,data_name):
        file = self._file(data_name)
        if os.path.isfile(file):
            with open(file,"rb") as f:
                print("Loading persistent model.")
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
    
    ##########################################
    def _has_persistent_model(self,data_name):
        return os.path.isfile(self._file(data_name))
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))
