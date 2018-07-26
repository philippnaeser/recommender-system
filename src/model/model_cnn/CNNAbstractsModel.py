# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:33:25 2018

@author: Steff
"""

import torch
import numpy as np

from AbstractClasses import AbstractModel

class CNNAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self):
        # number of recommendations to return
        self.recs = 10
    
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
        
        with torch.no_grad():
            scores = self.net.forward(vector)
            
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
        vectors = self.embeddings_parser.transform_vectors(batch)
    
        # pad inputs to max length
        max_len = max(len(l) for l in vectors)
        for i, inp in enumerate(vectors):
            vectors[i] = np.concatenate((inp,np.zeros(max_len-inp.size)))
            
        vectors = torch.FloatTensor(vectors).unsqueeze(1)
            
        with torch.no_grad():
            scores = self.net.forward(vectors)
            
        o = np.argsort(-scores.numpy())
        conference = list()
        confidence = list()
        index = 0
        for order in o:
            conference.append(
                    self.classes[order[0:self.recs]]
            )
            confidence.append(
                    scores[index,order[0:self.recs]]
            )
            index += 1
        
            
        return [conference,confidence]
        
    ##########################################
    def train(self,net,embeddings_parser,classes):
        self.net = net
        self.embeddings_parser = embeddings_parser
        self.classes = classes