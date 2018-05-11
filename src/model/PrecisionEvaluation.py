# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:55:39 2018

@author: Andreea
"""

from AbstractClasses import AbstractEvaluation

class PrecisionEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        """
        Calculates how many recommendations are relevant to the query
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: #recommended and attended conferences / #recommended conferences
            
        """
        countCorrectRecommendations = 0
        countRecommended = 0
        countNotInTest = 0
        
        allAttended = set()
        allRecommendations = set()
        
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
                
           
        ##Create a list of unique recommended conferences 
        for i in range(len(recommendation[0])):
            if recommendation[0][i] is not None:             
                for j in range(len(recommendation[0][i])):
                    if recommendation[0][i][j] is not None:  
                        allRecommendations.add(recommendation[0][i][j])
        
        #Count how many conferences from the test set were covered in the recommendations
        for conferences in allRecommendations:
            if conferences in allAttended:
                countCorrectRecommendations += 1
        
        
        ##Count how many conferences were recommended, including authors not
        ##in the test sets
        if recommendation[0] is not None:
            for rec in recommendation[0]:
                if rec is not None:
                    countRecommended += len(rec)
        
        
        ##Calculate precision (ignore authors who are not in the test set)
        if countNotInTest != countRecommended:
            measure = countCorrectRecommendations/(countRecommended-countNotInTest)
        else:
            measure = countCorrectRecommendations/countRecommended

        print("Precision = {}".format(measure))
        return measure
    