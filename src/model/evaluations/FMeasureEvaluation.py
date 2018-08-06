# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:12:50 2018

@author: Andreea
"""
import numpy as np

from AbstractClasses import AbstractEvaluation
from RecallEvaluation import RecallEvaluation
from PrecisionEvaluation import PrecisionEvaluation

class FMeasureEvaluation(AbstractEvaluation):
    
    def __init__(self,beta=1):
        self.beta = beta
    
    def evaluate(self, recommendation, truth):
        """
        Computes the F-beta measure of the model.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
            beta (int): Parameter which indicates how many times is recall
                        considered more important than precision
        
        Returns:
            int: F-beta score 
        """
        recall = RecallEvaluation().evaluate(recommendation, truth) 
        precision = PrecisionEvaluation().evaluate(recommendation, truth)
        
        if ((precision!=0) or (recall!=0)):
            measure = (1 + self.beta)*(precision*recall)/(np.square(self.beta)*precision + recall)
        else:
            measure = 0
        
        #print("F{} score = {}".format(beta, measure))
        return measure