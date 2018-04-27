# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:07:29 2018

@author: Andreea
"""
from AbstractClasses import AbstractEvaluation

class PrecisionAtKEvaluation(AbstractEvaluation):
    
    """
        Computes the precision at rank k
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
            k (int): the rank at which precision is calculated
        
        Returns:
            int: ratio of the number of recommended relevant conferences until 
                rank k and the total number of recommended conferences until
                rank k, normalized by the number of authors considered and who
                are in the test set
            
    """
    
    def evaluate(self, recommendation, truth, k):
        
        countRecUntilK = 0
        countNotInTest = 0
        countAuthors = len(recommendation[0])
        
        allAttended = set()
        topKRecommendations = set()
        
        ##Create a list of unique conferences from the test data
        ##Count how many authors are not in the test set
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    for j in range(len(truth[0][i])):
                        if truth[0][i][j] is not None:
                            allAttended.add(truth[0][i][j])
                else:
                    countNotInTest += 1
                i += 1
                
           
        ##Create a list of unique top k recommended conferences
        for i in range(len(recommendation[0])):
            if recommendation[0][i] is not None:             
                for j in range(len(recommendation[0][i])):
                    if recommendation[0][i][j] is not None:
                        if (j+1)<=k:
                            topKRecommendations.add(recommendation[0][i][j])
        
        #Count how many conferences from the test set were covered in the 
        #top k recommendations
        for conferences in topKRecommendations:
            if conferences in allAttended:
                countRecUntilK += 1
        
        ##Calculate precision at rank k 
        ##Normalize by the number of authors considered (ignore authors not in 
        ##the test set)
        if countAuthors!= countNotInTest:
            measure = countRecUntilK/(k*(countAuthors-countNotInTest))
        else:
            measure = 0
        
        print("Precision@{} = {}".format(k,measure))
        return measure
    
        
        
        
        