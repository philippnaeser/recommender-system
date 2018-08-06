# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:12:33 2018

@author: Steff
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(),".."))

query = [[
     ["A","B","C"]
     ,["A"]
     ,["A","B"]
     ,["A","B"]
     ,["A","B"]
     ,["C","D"]
     ,["A","A","B","A"]
     ,["A","B","A","C"]
     ,None
]]

truth = [[
    ["B"]
    ,["B"]
    ,["A"]
    ,["A","C"]
    ,["A","B","C","D"]
    ,["A","B","C","D"]
    ,["A"]
    ,["A","C"]
    ,["A"]
]]

# MeanRecall:
# ---------
# 1.0
# 0.0
# 1.0
# 0.5
# 0.5
# 0.5
# 1.0
# 1.0
# 0.0
# = 5.5/9 = 0.6111

# MeanPrecision(0):
# ---------
# 0.33
# 0.0
# 0.5
# 0.5
# 1.0
# 1.0
# 0.5
# 0.66
# 0.0 (!)
# = 4.5/9 = 0.5

# MeanPrecision(1):
# ---------
# 0.33
# 0.0
# 0.5
# 0.5
# 1.0
# 1.0
# 0.5
# 0.66
# 1.0 (!)
# = 5.5/9 = 0.611

# MeanF1Measure:
# ---------
# 0.5
# 0.0
# 0.667
# 0.5
# 0.667
# 0.667
# 0.667
# 0.795
# 0.0
# = 4.463/9 = 0.496

# MAP@10:
# ---------
# (0+1/2+0)/1 = 0.5
#               0.0
#               1.0
# (1+0)/2 =     0.5
# (1+1)/4 =     0.5
# (1+1)/4 =     0.5
#               1.0
# (1+3/4)/2 =   0.875
#               0.0
#----------------------
# = 4.875/9 = 0.542

# MAP@3:
# ---------
# (0+1/2+0)/1 = 0.5
#               0.0
#               1.0
# (1+0)/2 =     0.5
# (1+1)/4 =     0.5
# (1+1)/4 =     0.5
#               1.0
# (1)/2 =       0.5
#               0.0
#----------------------
# = 4.5/9 = 0.5

# MAP@1:
# ---------
#               0.0
#               0.0
#               1.0
#               0.5
#               0.5
#               0.5
#               0.25
# (1)/2 =       0.5
#               0.0
#----------------------
# = 3.25/9 = 0.361

from MeanRecallEvaluation import MeanRecallEvaluation
from MeanPrecisionEvaluation import MeanPrecisionEvaluation
from MeanFMeasureEvaluation import MeanFMeasureEvaluation
from MAPEvaluation import MAPEvaluation
from MAPkEvaluation import MAPkEvaluation

from EvaluationContainer import EvaluationContainer
evaluation = EvaluationContainer({
        "Recall":MeanRecallEvaluation(),
        "Precision0":MeanPrecisionEvaluation(0),
        "Precision1":MeanPrecisionEvaluation(1),
        "F1":MeanFMeasureEvaluation(),
        "MAP":MAPEvaluation(),
        "MAP@10":MAPkEvaluation(10),
        "MAP@3":MAPkEvaluation(3),
        "MAP@1":MAPkEvaluation(1)
})
evaluation.evaluate(query, truth)