# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:05:00 2018

@author: Steff
"""

import pandas as pd

class AbstractModel:
    
    ##########################################
    def query_single(self,q):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            q: The query as needed by the model.
        
        Returns:
            str[]: id of the conference
            str[]: name of the conference
            double[]: confidence scores
        """
        pass
    
    ##########################################
    def query_batch(self,batch):
        """
        Queries the model and returns a list of recommendations for each request.
        
        Args:
            batch[]: The list of queries as needed by the model.
        
        Returns:
            A list of size 'len(batch)' which contains the recommendations for each item of the batch.
            
            str[]: id of the conference
            str[]: name of the conference
            double[]: confidence scores
        """
        pass
    
    ##########################################
    def train(self,data):
        """
        Set the data to train the model. Will fail if 'data' is not a pandas DataFrame.
        
        Args:
            data (pandas.DataFrame): the data used by the model.
        """
        if not isinstance(data,pd.DataFrame):
            raise TypeError("argument 'data' needs to be of type pandas.DataFrame")
        
        pass
    
    ##########################################        
    def count_init(self,size,ticks=100):
        self.count_size = size
        self.count_i = 0
        self.count_checkpoint = max(int(size/ticks),1)
        
    ##########################################        
    def count(self):
        self.count_i += 1
        if (self.count_i % self.count_checkpoint == 0):
            print("Checkpoint reached: {}%".format(int(self.count_i*100/self.count_size)))



class AbstractEvaluation:
    
    def evaluate(self,recommendation,truth):
        return