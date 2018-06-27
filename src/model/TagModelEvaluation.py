# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

from TagModel import TagModel
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(".","..","data"))
from DataLoader import DataLoader

"""
    Prepare the data for the evaluation.
"""
#Training part
d = DataLoader()
d.papers(["2013","2014","2015"]).conferences().keywords()

data_train = d.data.loc[:, ["conference_name", "keyword_label"]]
data_train.columns = ["conference_name", "tag_name"]

model = TagModel()
model.train(data_train)

#Test Part
d = DataLoader()
d.papers(["2016"]).conferences().keywords()

data_test = d.data.loc[:, ["conference_name", "keyword_label"]]
data_test.columns = ["conference_name", "tag_name"]

model_test = TagModel()
model_test.train(data_test)


tags = list(data_test.tag_name)
print("Getting recommendations.")
recommendation = model.query_batch(tags)
print("Getting truth values.")
truth = model_test.query_batch(tags)

print("Computing FirstMatch.")
from FirstMatchEvaluation import FirstMatchEvaluation
evaluation = FirstMatchEvaluation()
evaluation.evaluate(recommendation,truth)

print("Computing recall.")
from RecallEvaluation import RecallEvaluation
evaluation = RecallEvaluation()
ev_recall = evaluation.evaluate(recommendation,truth)

print("Computing precision.")
from PrecisionEvaluation import PrecisionEvaluation
evaluation = PrecisionEvaluation()
ev_precision = evaluation.evaluate(recommendation,truth)

print("Computing F-measure.")
from FMeasureEvaluation import FMeasureEvaluation
evaluation = FMeasureEvaluation()
ev_f1 = evaluation.evaluate(recommendation,truth, 1)

print("Computing MAP.")
from MAPEvaluation import MAPEvaluation
evaluation = MAPEvaluation()
ev_map = evaluation.evaluate(recommendation, truth)

print("Computing Recall.")
from MeanRecallEvaluation import MeanRecallEvaluation
evaluation = MeanRecallEvaluation()
ev_mean_recall = evaluation.evaluate(recommendation, truth)

print("Computing Precision.")
from MeanPrecisionEvaluation import MeanPrecisionEvaluation
evaluation = MeanPrecisionEvaluation()
ev_mean_precision = evaluation.evaluate(recommendation, truth)

print("Recall: {}, Precision: {}, F1: {}, MAP: {}, Mean_Recall: {}, Mean_Precision: {}".format(ev_recall,ev_precision,ev_f1,ev_map,ev_mean_recall,ev_mean_precision))