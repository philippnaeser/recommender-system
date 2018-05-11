# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:09:45 2018

@author: Steff
@author: Andreea
"""

from AbstractClasses import AbstractEvaluation

class FirstMatchEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        """
        Checks if the first recommendation is part of the truth.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: numer of first recommendations that are part of the truth 
                normalized by the number of authors
            
        """
        count = 0
        countNotInTest = 0
        i = 0
        
        if truth[0] is not None:
            for conferences in truth[0]:
                if truth[0][i] is not None:
                    if recommendation[0][i] is not None: 
                        if recommendation[0][i][0] in truth[0][i]:
                            count += 1
                else:
                    countNotInTest += 1
                i += 1
        
        ##Calculate FirstMatch (ignore authors who are not in the test set)
        if countNotInTest != i:
            measure = count/(i-countNotInTest)
        else:
            measure = count/i
        
        print("FirstMatch = {}".format(measure))
        return measure