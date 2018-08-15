# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:58:43 2018

@author: Andreea
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(),".."))
from AbstractClasses import AbstractModel 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer # use the stemmer from nltk
from sklearn.preprocessing import LabelEncoder
import re
import pickle

class TFIDFClassifierAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,classifier,min_df=0,max_df=1.0,recs=10,ngram_range=(1,1),max_features=None,concat=True):
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stem_vectorizer = TfidfVectorizer(
                tokenizer=self
                ,stop_words="english"
                ,strip_accents = "unicode"
                ,min_df=min_df
                ,max_df=max_df
                ,ngram_range=ngram_range
                ,max_features=max_features
        )
        # number of recommendations to return
        self.recs = recs
        self.concat = concat
        self.classifier = classifier
        
        description_stem_matrix = "-".join([
                str(min_df),
                str(max_df),
                str(concat),
                str(ngram_range),
                str(max_features),
                "{}"
        ])
    
        description_classifier = "-".join([
                str(min_df),
                str(max_df),
                str(concat),
                str(ngram_range),
                str(max_features),
                classifier.__class__.__name__,
                "{}"
        ])
    
        self.path = os.path.join("..","..","..","data","processed","model_tfidf_classifier")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file_x = os.path.join(self.path,
                "stem."+description_stem_matrix+".pkl"
        )
        self.persistent_file_classifier = os.path.join(self.path,
                "classifier."+description_classifier+".pkl"
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
        self.count_init(len(batch))
        
        q_v = (self.stem_vectorizer.transform(batch))
        
        predicts = self.classifier.predict_proba(q_v)
        o = np.argsort(-np.array(predicts))
        print(o[:][0:self.recs])
        
        conferences = list()
        confidences = list()
        self.count_init(len(o))
        for index, order in enumerate(o):
            conferences.append(
                    self.labelencoder.inverse_transform(order[0:self.recs])
            )
            confidences.append(
                    predicts[index][order][0:self.recs]
            )
            self.count()
            
        return [conferences,confidences]
    
   ##########################################
    def train(self, data, data_name):
        if not self._load_model_x(data_name):
            print("Stem matrix not persistent yet. Creating now.")
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            if self.concat:
                # Concatenate abstracts per conferenceseries.
                data.chapter_abstract = data.chapter_abstract + " "
                data = data.groupby("conferenceseries").sum().reset_index()
            self.data = data
            
            # Generate stem matrix.
            self.stem_matrix = self.stem_vectorizer.fit_transform(self.data.chapter_abstract)
            self._save_model_x(data_name)
        
        if not self._load_model_classifier(data_name):
            print("Classifier not persistent yet. Creating now.")
            self.labelencoder = LabelEncoder()
            self.labels = self.labelencoder.fit_transform(self.data.conferenceseries)
            self.classifier.fit(self.stem_matrix,self.labels)
            self._save_model_classifier(data_name)
       
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
    def _file_classifier(self,data_name):
        return self.persistent_file_classifier.format(data_name)
    
    ##########################################
    def _save_model_x(self,data_name):
        file = self._file_x(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.stem_matrix, self.stem_vectorizer, self.data], f)
             
    ##########################################
    def _save_model_classifier(self,data_name):
        file = self._file_classifier(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.labelencoder, self.labels, self.classifier], f, protocol=4)
    
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
    def _load_model_classifier(self,data_name):
        file = self._file_classifier(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: Classifier")
            with open(file,"rb") as f:
                self.labelencoder, self.labels, self.classifier = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model(self,data_name):
        return self._load_model_x(data_name) &\
            self._load_model_classifier(data_name)
    
    ##########################################
    def _has_persistent_model(self,data_name):
        if os.path.isfile(self._file_x(data_name)) and\
            os.path.isfile(self._file_classifier(data_name)):
            return True
        return False
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))