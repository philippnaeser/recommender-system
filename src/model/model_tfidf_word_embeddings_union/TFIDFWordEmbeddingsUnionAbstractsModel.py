# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:40:34 2018

@author: Andreea
"""

from AbstractClasses import AbstractModel 
from EmbeddingsParser import EmbeddingsParser
from nltk.corpus import stopwords
import numpy as np
import os
import pickle
from scipy.spatial.distance import cdist


class TFIDFWordEmbeddingsUnionAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self, embedding_model, pretrained = True, concat=True, recs=10):
        self.stopList = stopwords.words('english') 
        self.embedding_model = embedding_model
        self.pretrained = pretrained
        
        self.parser = EmbeddingsParser()
        self.parser.load_model(self.embedding_model, self.pretrained)
        
        # number of recommendations to return
        self.recs = recs
        self.concat = concat
    
        description_embeddings = "-".join([
                str(concat),
                str(pretrained),
                str(self.embedding_model),
                "{}"
        ])
    
        self.path = os.path.join("..","..","..","data","processed","model_tfidfwordembeddings_union")
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
        q_v = self.parser.transform_tfidf_avg_vectors(
                self._remove_stopwords(batch),
                self.tfidf_weights
                )
        transformed_q_v = np.asarray(q_v)
        #print("Abstracts transformed.")
        #print("Dimensionality of batch: {}".format(transformed_q_v.shape))

        sim = 1-cdist(transformed_q_v, self.embedded_matrix, "cosine")
        #print("Cosine similarity computed.")
        o = np.argsort(-sim)

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
            print("Transformed data not persistent yet. Transforming now.")
            for check in ["chapter_abstract", "conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))

            data.chapter_abstract = self._remove_stopwords(data.chapter_abstract)
            if self.concat:
                data.chapter_abstract = data.chapter_abstract + " "
                data = data.groupby("conferenceseries").sum().reset_index()
                
            self.data = data
            self._save_model_x(data_name)
        else:
           if len(self.data) != len(data):
               raise ValueError("Mismatch vs. persistent training data size: Loaded: {} <-> Given: {}".format(len(self.data),len(data)))

        if not self._load_model_embeddings(data_name):
            print("Embeddings not persistent yet. Creating now.")
            self.tfidf_weights = self.parser.compute_tfidf_weights(
                                    self.data.chapter_abstract
                                    )
            self.embedded_matrix = self.parser.transform_tfidf_avg_vectors(
                    self.data.chapter_abstract,
                    self.tfidf_weights
                    )
            self.embedded_matrix = np.asarray(self.embedded_matrix)
            self._save_model_embeddings(data_name)

    ##########################################
    def _remove_stopwords(self, text):
        transformed_text = list()
        for sentence in text:
            content = " ".join([w for w in sentence.split() if w.lower() not in self.stopList])
            transformed_text.append(content)
        return transformed_text
    
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
            pickle.dump([self.stopList, self.data], f)
            
    ##########################################
    def _save_model_embeddings(self,data_name):
        file = self._file_embeddings(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.tfidf_weights, self.embedded_matrix], f)
    
    ##########################################
    def _load_model_x(self,data_name):
        file = self._file_x(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: X")
            with open(file,"rb") as f:
                self.stopList, self.data = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_model_embeddings(self,data_name):
        file = self._file_embeddings(data_name)
        if os.path.isfile(file):
            print("Loading persistent models: Embeddings")
            with open(file,"rb") as f:
                self.tfidf_weights, self.embedded_matrix = pickle.load(f)
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
