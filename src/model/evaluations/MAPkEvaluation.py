# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:29:06 2018

@author: Steff
"""

from AbstractClasses import AbstractEvaluation

class MAPkEvaluation(AbstractEvaluation):
    
    #########################################################
    def __init__(self,k=10,duplicates=True):
        self.k = k
        if duplicates:
            self.evaluate = self.evaluate_duplicates
        else:
            self.evaluate = self.evaluate_no_duplicates
    
    #########################################################
    def evaluate_no_duplicates(self,recommendation,truth):
        map_sum = 0
        for i, q in enumerate(recommendation[0]):
            hits = 0
            ap = 0
            denom = max(len(truth[0][i]),self.k)
            try:
                for rank, rec in enumerate(q[0:self.k],1):
                    add = rec in truth[0][i]
                    hits += add
                    ap += add*(hits/rank)
                map_sum += ap/denom
            except TypeError:
                pass # equivalent to adding 0 to map_sum if recommendation is None
        return map_sum/len(recommendation[0])
    
    #########################################################
    def evaluate_duplicates(self,recommendation,truth):
        map_sum = 0
        for i, q in enumerate(recommendation[0]):
            hits = 0
            ap = 0
            denom = min(len(truth[0][i]),self.k)
            hitlist = []
            try:
                for rank, rec in enumerate(q[0:self.k],1):
                    add = rec in truth[0][i]
                    hits += add
                    add = add & (rec not in hitlist)
                    if add: hitlist.append(rec)
                    ap += add*(hits/rank)
                #print("{} = {}".format(rec,ap/denom))
                map_sum += ap/denom
            except TypeError:
                pass # equivalent to adding 0 to map_sum if recommendation is None
        return map_sum/len(recommendation[0])