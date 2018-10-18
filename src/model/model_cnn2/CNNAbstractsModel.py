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
        os.path.realpath(__file__),"..","..")
)
from AbstractClasses import AbstractModel
from CNNet2 import CNNet2
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","..","data")
)
from EmbeddingsParser import EmbeddingsParser

class CNNAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,net_name,epoch=None,recs=10):
        # number of recommendations to return
        self.recs = recs
        
        # load the network from disk
        self.net = CNNet2.from_disk(net_name,epoch=epoch)
        self.softmax = torch.nn.Softmax(dim=1)
        self.embeddings_parser = EmbeddingsParser()
        self.embeddings_parser.load_model(self.net.embedding_model)
        self.embeddings_size = EmbeddingsParser.lengths[self.net.embedding_model]
        self.spatial_size = 300
    
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
        #print("transforming")
        
        vectors = self.embeddings_parser.transform_tensor_to_fixed_size(
                batch,embeddings_size=self.embeddings_size,spatial_size=self.spatial_size
        )
        del batch
      
        #print("inputs")
        
        inputs = torch.FloatTensor(vectors)
        del vectors
            
        #print("forward")
        
        with torch.no_grad():
            scores = self.softmax(self.net.forward(inputs))
            
        del inputs
        
        #print("recs")
            
        o = np.argsort(-scores.numpy())
        conference = list()
        confidence = list()
        index = 0
        for index, order in enumerate(o):
            conference.append(
                    self.net.classes[order[0:self.recs]].tolist()
            )
            confidence.append(
                    scores[index,order[0:self.recs]].tolist()
            )        
            
        return [conference,confidence]
        
    ##########################################
    def train(self):
        pass