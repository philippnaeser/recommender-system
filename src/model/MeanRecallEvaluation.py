# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:31:50 2018

@author: Steff
"""

from AbstractClasses import AbstractEvaluation

class MeanRecallEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        """
        Calculates the fraction of attended conferences that were recommended.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: #recommended and attended conferences / #attended conferences
            
        """
        recall = 0
        size = len(recommendation[0])
        for i in range(size):
            q_s = set(recommendation[0][i])
            t_s = set(truth[0][i])
            recall += len(q_s.intersection(t_s)) / len(t_s)
            
        return recall/size