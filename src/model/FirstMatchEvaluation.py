# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:09:45 2018

@author: Steff
@author: Andreea
"""

from AbstractClasses import AbstractEvaluation

class FirstMatchEvaluation(AbstractEvaluation):
    
    """
        Checks if the first recommendation is part of the truth.
        
        Args:
            recommendation (list): The list of recommendations returned 
                                    by the model.
            truth (list): The test set list of conferences attended
        
        Returns:
            int: 1, if first recommendation is part of the truth
                 0, if the first recommendation is not part of the truth
                -1, if the query is a single author who is not covered in the 
                test set
            
    """
    
    def evaluate(self,recommendation,truth):
        
        count = 0
        i = 0
        
        if not all (attended is None for attended in truth[0]):
            for conferences in truth[0]:
                ##Check if the author is in the test data
                if truth[0][i] is None:
                    pass
                else:
                    if recommendation[0][i] is not None: 
                        if recommendation[0][i][0] in conferences[0]:
                            count += 1
                    i += 1
        
        if i!=0:
            measure = count/i
        else:
            measure = -1
            print("The author is not in the test set")

        print("FirstMatch = {}".format(measure))
        return measure