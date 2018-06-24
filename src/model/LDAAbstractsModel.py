# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:17:57 2018

@author: Andreea
"""

from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemmer from nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import re
import os
import pickle

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:17:57 2018

@author: Andreea
"""


class LDAAbstractsModel(AbstractModel):
    
    persistent_file_x = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","processed","abstracts.lda.model.X.pkl"
    )
    persistent_file_factors = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","processed","abstracts.lda.model.factors.pkl"
    )
    
    ##########################################
    def __init__(self,recs=10):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,strip_accents = "unicode"
                ,min_df=10
                ,max_df=0.6
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
        transformed_q_v = (self.lda.transform(q_v))

        sim = cosine_similarity(transformed_q_v,self.transformed_matrix)[0]
        o = np.argsort(-sim)
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
        transformed_q_v = (self.lda.transform(q_v))
        print("Abstracts transformed.")
        print(q_v.shape)
        print(transformed_q_v.shape)
        
        sim = cosine_similarity(transformed_q_v, self.transformed_matrix)
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
    def train(self, data, dimensions):
        if not self._load_model_x():
            print("Stem matrix not persistent yet. Creating now.")
            for check in ["chapter_abstract","conference","conference_name"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            self.data = data
            self.stem_matrix = self.stem_vectorizer.fit_transform(data.chapter_abstract.str.decode("unicode_escape"))
            self._save_model_x()
            
        if not self._load_model_factors():
            print("LDA not persistent yet. Creating now.")
            self.dimensions = dimensions
            self.lda = LatentDirichletAllocation(
                    n_components=self.dimensions
                    ,verbose = 1
                    ,learning_method = 'online'
                    ,random_state=0)
            self.lda.fit(self.stem_matrix)
            self.transformed_matrix = self.lda.transform(self.stem_matrix)
            
            self._save_model_factors()
        
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
        with open(LDAAbstractsModel.persistent_file_x,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
            
    ##########################################
    def _save_model_factors(self):
        with open(LDAAbstractsModel.persistent_file_factors,"wb") as f:
            pickle.dump([self.lda, self.transformed_matrix], f)
    
    ##########################################
    def _load_model_x(self):
        if os.path.isfile(LDAAbstractsModel.persistent_file_x):
            print("Loading persistent models: X")
            with open(LDAAbstractsModel.persistent_file_x,"rb") as f:
                self.stem_matrix, self.stem_vectorizer, self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_factors(self):
        if os.path.isfile(LDAAbstractsModel.persistent_file_factors):
            print("Loading persistent models: Factors")
            with open(LDAAbstractsModel.persistent_file_factors,"rb") as f:
                self.lda, self.transformed_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))