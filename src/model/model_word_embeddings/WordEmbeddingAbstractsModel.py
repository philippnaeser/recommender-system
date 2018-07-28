# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:40:34 2018

@author: Andreea
"""

from AbstractClasses import AbstractModel 
from nltk.corpus import stopwords
import numpy as np
import os
import pickle
from EmbeddingsParser import EmbeddingsParser
from scipy.spatial.distance import cdist


class WordEmbeddingAbstractsModel(AbstractModel):
    
    persistent_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","..","data","processed","abstracts.wordembbeding.model.pkl"
    )

    ##########################################
    def __init__(self, pretrained = True, recs=10):
        self.stopList = stopwords.words('english')  
        self.parser = EmbeddingsParser()
        self.pretrained = pretrained
        
        # number of recommendations to return
        self.recs = recs
    
    ##########################################
    def query_single(self, abstract):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            abstract (string): The abstract as a string.
        
        Returns:
            str[]: name of the conference
            double[]: confidence scores
        """
        q_v = self.parser.transform_avg_vectors(self._remove_stopwords([abstract]))
        transformed_q_v = np.asarray(q_v)
        sim = 1-cdist(transformed_q_v, self.embedded_matrix, "cosine")
        o = np.argsort(-sim)
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
        q_v = self.parser.transform_avg_vectors(self._remove_stopwords(batch))
        transformed_q_v = np.asarray(q_v)
        print("Abstracts transformed.")
   
        sim = 1-cdist(transformed_q_v, self.embedded_matrix, "cosine")
        o = np.argsort(-sim)

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
    def train(self, data, embedding_model):
        if not self._load_model():
            print("Embedded matrix not persistent yet. Creating now.")
            for check in ["chapter_abstract","conference","conference_name"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
                    
            self.data = data
            self.parser.load_model(embedding_model, self.pretrained)
            self.embedded_matrix = self.parser.transform_avg_vectors(
                    self._remove_stopwords(
                            data.chapter_abstract.str.decode("unicode_escape")
                            )
                    )
            self.embedded_matrix = np.asarray(self.embedded_matrix)
            self._save_model()
            
        else:
            self.parser.load_model(embedding_model)
  
    ##########################################
    def _remove_stopwords(self, text):
        transformed_text = list()
        for sentence in text:
            content = " ".join([w for w in sentence.split() if w.lower() not in self.stopList])
            transformed_text.append(content)
        return transformed_text
    
    ##########################################
    def _save_model(self):
        with open(WordEmbeddingAbstractsModel.persistent_file, "wb") as f:
            pickle.dump([self.stopList, self.data, self.embedded_matrix], f)
            
    
    ##########################################
    def _load_model(self):
        if os.path.isfile(WordEmbeddingAbstractsModel.persistent_file):
            print("Loading persistent models: Embedded matrix")
            with open(WordEmbeddingAbstractsModel.persistent_file,"rb") as f:
                self.stopList, self.data, self.embedded_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
