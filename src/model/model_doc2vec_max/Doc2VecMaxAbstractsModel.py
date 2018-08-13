# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 00:39:29 2018

@author: Andreea
"""

from AbstractClasses import AbstractModel 
from Doc2VecParser import Doc2VecParser
from scipy.spatial.distance import cdist
import numpy as np
import os
import pickle

class Doc2VecMaxAbstractsModel(AbstractModel):
    
    
    ##########################################
    def __init__(self, embedding_model, recs=10):
        self.embedding_model = embedding_model
        self.parser = Doc2VecParser()
        self.parser.load_model(self.embedding_model)

        # number of recommendations to return
        self.recs = recs
    
        description_embeddings = "-".join([
                str(self.embedding_model),
                "{}"
        ])
    
        self.path = os.path.join("..","..","..","data","processed","model_doc2vec_max")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file_x = os.path.join(self.path,
                                              "model-X.pkl")
        
        self.persistent_file_embeddings = os.path.join(self.path,
                                               "model-"+description_embeddings+"-Embeddings.pkl")
    
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
        
        q_v = self.parser.transform_vectors(batch)
        transformed_q_v = np.asarray(q_v)
        #print("Abstracts transformed.")
        #print("Dimensionality of batch: {}".format(transformed_q_v.shape))
        
        sim = 1-cdist(transformed_q_v, self.embedded_matrix, "cosine")
        #print("Cosine similarity computed.")
        o = np.argsort(-sim)

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
    def train(self, data, data_name):
        if not self._load_model_x(data_name):
            print("Transformed data not persistent yet. Transforming now.")
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))

            self.data = data
            self._save_model_x(data_name)
        else:
           if len(self.data) != len(data):
               raise ValueError("Mismatch vs. persistent training data size: Loaded: {} <-> Given: {}".format(len(self.data),len(data)))

        if not self._load_model_embeddings(data_name):
            print("Embeddings not persistent yet. Creating now.")
            self.embedded_matrix = self.parser.transform_vectors(data.chapter_abstract)
            self.embedded_matrix = np.asarray(self.embedded_matrix)
            self._save_model_embeddings(data_name)
        
    ##########################################
    def _file_x(self,data_name):
        return self.persistent_file_x.format(data_name)
    
    ##########################################
    def _file_embeddings(self,data_name):
        return self.persistent_file_embeddings.format(data_name)
    
    ##########################################
    def _save_model_x(self,data_name):
        file = self._file_x(data_name)
        with open(file,"wb") as f:
            pickle.dump(self.data, f)
            
    ##########################################
    def _save_model_embeddings(self,data_name):
        file = self._file_embeddings(data_name)
        with open(file,"wb") as f:
            pickle.dump(self.embedded_matrix, f)
    
    ##########################################
    def _load_model_x(self,data_name):
        file = self._file_x(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: X")
            with open(file,"rb") as f:
                self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_embeddings(self,data_name):
        file = self._file_embeddings(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: Embeddings")
            with open(file,"rb") as f:
                self.embedded_matrix = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model(self,data_name):
        return self._load_model_x(data_name) & self._load_model_embeddings(data_name)

    
    ##########################################
    def _has_persistent_model(self,data_name):
        if os.path.isfile(self._file_x(data_name)) and os.path.isfile(self._file_embeddings(data_name)):
            return True
        return False