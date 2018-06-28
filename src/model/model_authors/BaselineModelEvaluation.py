# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:23:29 2018

@author: Steff
"""
import sys
import os
sys.path.insert(0, os.path.join(".","..","..","data"))
sys.path.insert(0, os.path.join(".",".."))
sys.path.insert(0, os.path.join(".","..","evaluations"))

from BaselineModel import BaselineModel
from DataLoader import DataLoader
#import pandas as pd
#import numpy as np

"""
    Prepare the data for the evaluation.
"""

d = DataLoader()
d.papers(["2013","2014","2015"]).contributions().conferences()
data_train = d.data

model = BaselineModel()
model.train(data_train)

d = DataLoader()
d.papers(["2016"]).contributions().conferences()
data_test = d.data

model_test = BaselineModel()
model_test.train(data_test)

authors = list(model_test.get_author_names(count=0))

print("Getting recommendations.")
recommendation = model.query_batch(authors)
print("Getting truth values.")
truth = model_test.query_batch(authors)

"""
    Get some evaluation scores.
    
    Structure of parameters (for authors = [Heiko Paulheim, Sven Hertling]):
        recommendation/truth = [
                [
                        ['International Semantic Web Conference',
                         'Extended Semantic Web Conference',
                         'European Semantic Web Conference',
                         'International Conference on Discovery Science'
                         ],
                        ['Extended Semantic Web Conference']
                ],
                [
                        [6.0, 3.0, 2.0, 1.0],
                        [1.0]
                ]
        ]
"""

# add your evaluations here
# call your file <yourname>Evaluation.py

#print("Computing FirstMatch.")
#from FirstMatchEvaluation import FirstMatchEvaluation
#evaluation = FirstMatchEvaluation()
#evaluation.evaluate(recommendation,truth)

#print("Computing recall.")
#from RecallEvaluation import RecallEvaluation
#evaluation = RecallEvaluation()
#evaluation.evaluate(recommendation,truth)

#print("Computing precision.")
#from PrecisionEvaluation import PrecisionEvaluation
#evaluation = PrecisionEvaluation()
#evaluation.evaluate(recommendation,truth)

#print("Computing F-measure.")
#from FMeasureEvaluation import FMeasureEvaluation
#evaluation = FMeasureEvaluation()
#evaluation.evaluate(recommendation,truth, 1)

print("Computing MAP.")
from MAPEvaluation import MAPEvaluation
evaluation = MAPEvaluation()
ev_map = evaluation.evaluate(recommendation, truth)

print("Computing Recall.")
from MeanRecallEvaluation import MeanRecallEvaluation
evaluation = MeanRecallEvaluation()
ev_recall = evaluation.evaluate(recommendation, truth)

print("Computing Precision.")
from MeanPrecisionEvaluation import MeanPrecisionEvaluation
evaluation = MeanPrecisionEvaluation()
ev_precision = evaluation.evaluate(recommendation, truth)

print("Computing F1Measure.")
from MeanFMeasureEvaluation import MeanFMeasureEvaluation
evaluation = MeanFMeasureEvaluation()
ev_fmeasure = evaluation.evaluate(recommendation, truth, 1)

print("Recall: {}, Precision: {}, F1Measure: {}, MAP: {}".format(ev_recall, ev_precision, ev_fmeasure, ev_map))