# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:58:50 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation
from PrecisionAtKEvaluation import PrecisionAtKEvaluation

class AveragePrecisionEvaluation(AbstractEvaluation):
    
    """
        Computes the average precision for a query (i.e. the average of the 
        Precision@K scores for all the ranks of the recommended and attended 
        conferences)
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: the average of all the precision@k values for the top 
                k conferences that are correctly recommended
            
    """
    
    def evaluate(self,recommendation,truth):
        
        rank = 0
        countAttended = 0
        sumPrecisions = 0
        
        allAttended = set()
        
        ##Create a list of unique conferences from the test data
        ##Count how many authors are not in the test set
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    for j in range(len(truth[0][i])):
                        if truth[0][i][j] is not None:
                            allAttended.add(truth[0][i][j])
                i += 1
        
        ##Sum the precision@k for the correct recommendations
        for i in range(len(recommendation[0])):
            if recommendation[0][i] is not None:             
                for j in range(len(recommendation[0][i])):
                    if recommendation[0][i][j] is not None:
                        if recommendation[0][i][j] in allAttended:
                            rank = j+1
                            sumPrecisions += PrecisionAtKEvaluation().evaluate(recommendation,truth,rank)
        
        
        ##Count how many conferences the author actually attended 
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    countAttended += len(truth[0][i])
                i += 1
        
        ##Calculate the average precision for the query
        if countAttended!=0:
            measure = sumPrecisions/countAttended
        else:
            measure = 0
        
        print("Average precision = {}".format(measure))
        return measure
                            
        
        