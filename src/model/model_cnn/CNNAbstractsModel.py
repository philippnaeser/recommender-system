# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:33:25 2018

@author: Steff
"""

import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..")
)
from AbstractClasses import AbstractModel
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","neuralnets")
)
from CNNet import CNNet
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","..","data")
)
from EmbeddingsParser import EmbeddingsParser

class CNNAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,net_name,recs=10):
        # number of recommendations to return
        self.recs = recs
        
        # load the network from disk
        self.net = CNNet.from_disk(net_name)
        self.embeddings_parser = EmbeddingsParser()
        self.embeddings_parser.load_model(self.net.embedding_model)
    
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
        vector = self.embeddings_parser.transform_vector(abstract)
        input = torch.FloatTensor(vector).unsqueeze(0).unsqueeze(0)
        del vector
        
        with torch.no_grad():
            scores = self.net.forward(input)
            
        del input
        
        order = np.argsort(-scores.numpy())[0]
        conference = self.net.classes[order[0:self.recs]]
        confidence = scores[0,order[0:self.recs]]
            
        return [conference,confidence.numpy()]
            
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
        print("transforming")
        
        vectors = self.embeddings_parser.transform_vectors(batch)
        del batch
        
        print("padding")
    
        # pad inputs to max length
        max_len = max(len(l) for l in vectors)
        for i, inp in enumerate(vectors):
            vectors[i] = np.concatenate((inp,np.zeros(max_len-inp.size)))
        
        print("inputs")
        
        inputs = torch.FloatTensor(vectors).unsqueeze(1)
        del vectors
            
        print("forward")
        
        with torch.no_grad():
            scores = self.net.forward(inputs)
            
        del inputs
        
        print("recs")
            
        o = np.argsort(-scores.numpy())
        conference = list()
        confidence = list()
        index = 0
        for order in o:
            conference.append(
                    self.net.classes[order[0:self.recs]]
            )
            confidence.append(
                    scores[index,order[0:self.recs]]
            )
            index += 1
        
            
        return [conference,confidence]
        
    ##########################################
    def train(self):
        pass