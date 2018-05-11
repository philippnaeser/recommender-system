# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:57:56 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation
from PrecisionAtKEvaluation import PrecisionAtKEvaluation

class RPrecisionEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        """
        Computes the R-precision of a query
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: the precision at rank k, where k is equal to the number of
                relevant conferences (from the ground truth) for the query
            
        """
        
        rank = 0
        
        ##Count how many conferences the author actually attended 
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    rank += len(truth[0][i])
                i += 1
        
        ##Calculate R-Precision 
        measure = PrecisionAtKEvaluation().evaluate(recommendation,truth,rank)
        
        print("R-Precision = {}".format(measure))
        return measure    