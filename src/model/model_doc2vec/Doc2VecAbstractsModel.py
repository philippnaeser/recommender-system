# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:41:59 2018

@author: Andreea
"""


from AbstractClasses import AbstractModel 
from Doc2VecParser import Doc2VecParser
from scipy.spatial.distance import cdist
import numpy as np
import os
import pickle

class Doc2VecAbstractsModel(AbstractModel):
    
    persistent_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","..","data","processed","abstracts.doc2vec.model.pkl"
    )
    
    ##########################################
    def __init__(self, recs=10):
        self.parser = Doc2VecParser()
        
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
        q_v = self.parser.transform_vectors([abstract])
        transformed_q_v = np.asarray(q_v)
        
        sim = 1-cdist(transformed_q_v, self.embedded_matrix, "cosine")
        o = np.argsort(-sim)
        
        return [
                list(self.data.iloc[o][0:self.recs].conferenceseries),
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
        print("Transforming abstracts.")
        q_v = self.parser.transform_vectors(batch)
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
                    list(self.data.iloc[order][0:self.recs].conferenceseries)
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
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
                    
            self.parser.load_model(embedding_model)
            self.data = data
            self.embedded_matrix = self.parser.transform_vectors(data.chapter_abstract)
            self.embedded_matrix = np.asarray(self.embedded_matrix)
            
            self._save_model()
        else:
            self.parser.load_model(embedding_model)

    
    ##########################################
    def _save_model(self):
        with open(Doc2VecAbstractsModel.persistent_file, "wb") as f:
            pickle.dump([self.data, self.embedded_matrix], f)
            
    
    ##########################################
    def _load_model(self):
        if os.path.isfile(Doc2VecAbstractsModel.persistent_file):
            print("Loading persistent models: Embedded matrix")
            with open(Doc2VecAbstractsModel.persistent_file,"rb") as f:
                self.data, self.embedded_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False

