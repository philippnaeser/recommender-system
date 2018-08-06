# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:17:23 2018

@author: Steff
"""
class EvaluationContainer():
    
    def __init__(self,evaluations=None):
        if evaluations is not None:
            self.evaluations = evaluations
        else:
            # Create a standard repertoire.
            from MeanRecallEvaluation import MeanRecallEvaluation
            from MeanPrecisionEvaluation import MeanPrecisionEvaluation
            from MAPEvaluation import MAPEvaluation
            self.evaluations = {
                "Recall":MeanRecallEvaluation(),
                "Precision1":MeanPrecisionEvaluation(1),
                "Precision0":MeanPrecisionEvaluation(0),
                "MAP":MAPEvaluation()
            }
        
        self.max_len = str(len(max(self.evaluations, key=len))+1)
    
    def evaluate(self,rec,truth):
        results = []
        
        for name, e in self.evaluations.items():
            result = e.evaluate(rec,truth)
            result_rounded = round(result,3)
            results.append(result)
            print(("{:<"+self.max_len+"s}= {:0.3f}").format(name,result_rounded))
            
        print(" ".join([str(r) for r in results]))
        return results