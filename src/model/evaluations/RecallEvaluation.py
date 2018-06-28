# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:09:45 2018

@author: Steff
@author: Andreea
"""


from AbstractClasses import AbstractEvaluation

class RecallEvaluation(AbstractEvaluation):
    
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
        countCorrectRecommendations = 0
        countAttended = 0
        
        allAttended = set()
        allRecommendations = set()
        
        ##Create a list of unique conferences from the test data
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    for j in range(len(truth[0][i])):
                        if truth[0][i][j] is not None:
                            allAttended.add(truth[0][i][j])
                            
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
        
        ##Count how many conferences the author actually attended 
        i = 0
        if truth[0] is not None:
            for attended in truth[0]:
                if truth[0][i] is not None:
                    countAttended += len(truth[0][i])
                i += 1
        
        #Calculate recall
        ##Set recall to 1 if there are no attended conferences in the test set
        if countAttended!=0:
            measure = countCorrectRecommendations/countAttended
        else:
            measure = 1
        
        
        print("Recall = {}".format(measure))
        return measure