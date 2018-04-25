# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:09:45 2018

@author: Steff
@author: Andreea
"""


from AbstractClasses import AbstractEvaluation

class RecallEvaluation(AbstractEvaluation):
    
    """
        Calculates how many conferences from the test set were covered in 
        the recommendations.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: #recommendedConferences / #attendedConferences, or
                -1, if the query is a single author who is not covered in the 
                test set
            
    """
    
    def evaluate(self,recommendation,truth):       
        
        countRecommended = 0
        countAttended = 0
        
        allAttended = set()
        allRecommendations = set()
        
        ##Create a list of unique conferences from the test data
        i = 0
        if not all (attended is None for attended in truth[0]):
            for conferences in truth[0]:   
                ##Check if the author is in the test data
                if truth[0][i] is None:
                    pass
                else:
                    allAttended.add(conferences[0])
                i += 1
        
        
        ##Create a list of unique recommended conferences 
        for i in range(len(recommendation[0])):
            if recommendation[0][i] is not None:             
                for j in range(len(recommendation[0][i])):
                    if recommendation[0][i][j] is not None:  
                        allRecommendations.add(recommendation[0][i][j])
        
        #Count how many conferences from the test set were covered in the recommendations
        for recommendation in allRecommendations:
            if recommendation in allAttended:
                countRecommended += 1
        
        ##Count how many conferences the author actually attended 
        k = 0
        if not all (attended is None for attended in truth[0]):
            for conferences in truth[0]: 
                ##Check if the author is in the test data
                if truth[0][k] is None:
                    pass
                else:
                    countAttended += len(truth[0][k])
                k += 1                                   
             
        if countAttended!=0:
            measure = countRecommended/countAttended
        else:
            measure = -1
            print("The author is not in the test set")

        print("Recall = {}".format(measure))
        return measure
    