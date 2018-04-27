# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:59:26 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation
from AveragePrecisionEvaluation import AveragePrecisionEvaluation

class MAPEvaluation(AbstractEvaluation):
    
    """
        Computes the mean average precision (MAP) for all queries 
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: the mean of all average precision scores for all the queries
            
    """
    
    def evaluate(self,recommendation,truth):
        
        sumAveragePrecision = 0
        i = 0
        
        ##Sum the average precision values for all the queries
        if recommendation[0]:
            for conference in recommendation[0]:
                if conference is not None:
                    sumAveragePrecision += AveragePrecisionEvaluation().evaluate(recommendation, truth)
                i += 1
        
        ##Calculate the mean average precision
        if i!= 0:
            measure = sumAveragePrecision/i
        else:
            measure = 0
        
        print("Mean average precision = {}".format(measure))
        return measure
                
                
            