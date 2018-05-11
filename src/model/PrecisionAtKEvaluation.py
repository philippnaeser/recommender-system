# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:07:29 2018

@author: Andreea
"""
from AbstractClasses import AbstractEvaluation

class PrecisionAtKEvaluation(AbstractEvaluation):
    
    def evaluate(self, recommendation, truth, k):
        """
        Computes the precision at rank k for a single query.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
            k (int): the rank at which precision is calculated
        
        Returns:
            int: ratio of the number of recommended relevant conferences until 
                rank k and the total number of recommended conferences until
                rank k
            
        """ 
        countRelevantRetrieved = 0   
        
        for i in range(len(recommendation[0])):
            if truth[0][i] is not None:
                if recommendation[0][i] is not None:  
                    for j in range(len(recommendation[0][i])):
                        rank = j+1
                        if (recommendation[0][i][j] is not None) and (rank<=k):
                            if recommendation[0][i][j] in truth[0][i]:
                                countRelevantRetrieved += 1
        
            if k!= 0:
                measure = countRelevantRetrieved/k
            else:
                measure = 0
         
        #print("Precision@{} = {}".format(k,measure))
        return measure
    
        
        
        
        