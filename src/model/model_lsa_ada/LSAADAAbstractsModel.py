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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
import pickle

class LSAADAAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,recs=10,dimensions=1000,min_df=3,max_df=1.0):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,strip_accents = "unicode"
                ,min_df=min_df
                ,max_df=max_df
        )
        
        persistent_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..","..","..","data","processed","model_lsa_ada"
        )
        
        if not os.path.isdir(persistent_path):
            os.mkdir(persistent_path)
        
        model_description = str(dimensions) + "." + str(min_df) + "." + str(max_df)
        
        self.persistent_file_x = os.path.join(
                persistent_path,
                "abstracts.lsa.ada.model."+model_description+".X.pkl"
        )
        self.persistent_file_factors = os.path.join(
                persistent_path,
                "abstracts.lsa.ada.model."+model_description+".factors.pkl"
        )
        self.persistent_file_ada = os.path.join(
                persistent_path,
                "abstracts.lsa.ada.model."+model_description+".ada.pkl"
        )
        
        # number of recommendations to return
        self.recs = recs
        self.ada = AdaBoostClassifier()
    
    
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
        pass
    
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
        transformed_q_v = (self.trsvd.transform(q_v))
        print("Abstracts transformed.")
        print(q_v.shape)
        print(transformed_q_v.shape)

        test_p = self.ada.predict(transformed_q_v)
        print(test_p)
    
        predicts = self.ada.predict_proba(transformed_q_v)
        o = np.argsort(-np.array(predicts))
        print(o[:][0:self.recs])
        
        conference = list()
        confidence = list()
        self.count_init(len(o))
        for index, order in enumerate(o):
            conference.append(
                    self.labelencoder.inverse_transform(order[0:self.recs])
            )
            confidence.append(
                    predicts[index][order][0:self.recs]
            )
            self.count()
        
            
        return [conference,confidence]
    
   ##########################################
    def train(self, data):
        if not self._load_model_x():
            print("Stem matrix not persistent yet. Creating now.")
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract)
            self._save_model_x()
            
        if not self._load_model_factors():
            print("SVD not persistent yet. Creating now.")
            self.trsvd = TruncatedSVD(n_components=self.dimensions, random_state=0)
            self.transformed_matrix = self.trsvd.fit_transform(self.stem_matrix)
            self._save_model_factors()
            
        if not self._load_model_ada():
            self.labelencoder = LabelEncoder()
            self.labels = self.labelencoder.fit_transform(data.conferenceseries)
            self.ada.fit(self.transformed_matrix,self.labels)
            self._save_model_ada()
        
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
        with open(self.persistent_file_x,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
            
    ##########################################
    def _save_model_factors(self):
        with open(self.persistent_file_factors,"wb") as f:
            pickle.dump([self.trsvd, self.transformed_matrix], f)
            
    ##########################################
    def _save_model_ada(self):
        with open(self.persistent_file_ada,"wb") as f:
            pickle.dump([self.labelencoder, self.labels, self.ada], f)
    
    ##########################################
    def _load_model_x(self):
        if os.path.isfile(self.persistent_file_x):
            print("Loading persistent models: X")
            with open(self.persistent_file_x,"rb") as f:
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_factors(self):
        if os.path.isfile(self.persistent_file_factors):
            print("Loading persistent models: Factors")
            with open(self.persistent_file_factors,"rb") as f:
                self.trsvd, self.transformed_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False

    ##########################################
    def _load_model_ada(self):
        if os.path.isfile(self.persistent_file_ada):
            print("Loading persistent models: ADA")
            with open(self.persistent_file_ada,"rb") as f:
                self.labelencoder, self.labels, self.ada = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))