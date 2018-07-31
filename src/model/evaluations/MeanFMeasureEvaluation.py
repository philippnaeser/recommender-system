# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:34:56 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation
import numpy as np

class MeanFMeasureEvaluation(AbstractEvaluation):
    
    def evaluate(self, recommendation, truth, beta):
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
        
        precision = 0
        recall = 0
        fmeasure = 0
        
        size = len(recommendation[0])
        for i in range(size):
            if recommendation[0][i] is None:
                precision += 1
                recall += 0
            elif truth[0][i] is None:
                precision += 0
                recall += 1
            else:
                q_s = set(recommendation[0][i])
                t_s = set(truth[0][i])
                precision = len(q_s.intersection(t_s)) / len(q_s)
                recall = len(q_s.intersection(t_s)) / len(t_s)
                
                if (precision != 0) & (recall != 0):
                    fmeasure += (1 + beta)*(precision*recall)/(np.square(beta)*precision + recall)
                else:
                    fmeasure += 0
                
        return fmeasure/size
        