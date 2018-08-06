# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:14:36 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation

class MAPEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        """
        Computes the mean average precision (MAP) for all queries. 
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: the mean of all average precision scores for all the queries
            
        """ 
        sumAveragePrecisions = 0
        rank = 0
        
        #Sum the average precisions@k for all queries
        for i in range(len(recommendation[0])):
           
            sumPrecisions = 0
            averagePrecision = 0
              
            if truth[0][i] is not None:
                if recommendation[0][i] is not None: 
                    checked = set()
                    
                    for j in range(len(recommendation[0][i])):
                        if (recommendation[0][i][j] is not None) and (recommendation[0][i][j] in truth[0][i]) and not(recommendation[0][i][j] in checked):
                           rank = j+1
                           sumPrecisions = sumPrecisions + self._precisionAtK(recommendation[0][i], truth[0][i], rank)
                           checked.add(recommendation[0][i][j])
                           
                    averagePrecision = sumPrecisions/len(truth[0][i])
            sumAveragePrecisions += averagePrecision

        measure = sumAveragePrecisions/len(recommendation[0])
        
        #print("Mean average precision = {}".format(measure))
        return measure
    
    
    def _precisionAtK(self, recommendation, truth, k):
        
        countRelevantRetrieved = 0   
        
        for i in range(len(recommendation)):
            rank = i+1
            if (recommendation[i] in truth) and (rank<=k):
                countRelevantRetrieved += 1

        if k!= 0:
            measure = countRelevantRetrieved/k
        else:
            measure = 0
        
        return measure
