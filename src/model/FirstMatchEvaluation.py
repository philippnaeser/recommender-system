# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:09:45 2018

@author: Steff
"""

from AbstractClasses import AbstractEvaluation

class FirstMatchEvaluation(AbstractEvaluation):
    
    def evaluate(self,recommendation,truth):
        count = 0
        i = 0
        
        for conferences in truth[0]:
            if recommendation[0][i] is not None:
                if recommendation[0][i][0] in conferences[0]:
                    count += 1
            i += 1
        
        measure = count/i
        print("FistMatch = {}".format(measure))
        return count/i