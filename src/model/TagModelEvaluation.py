# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

from TagModel import TagModel
import pandas as pd
import os

"""
    Prepare the data for the evaluation.
"""
#Training part
data_train = pd.read_csv(os.path.join("..","..","data","processed","tags.csv"))

model = TagModel()
model.train(data_train)


data_test = pd.read_csv(os.path.join("..","..","data","processed","tags_test.csv"))
model_test = TagModel()
model_test.train(data_test)

tags = list(model_test.get_tag_names(count=0))
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
evaluation.evaluate(recommendation,truth)

print("Computing precision.")
from PrecisionEvaluation import PrecisionEvaluation
evaluation = PrecisionEvaluation()
evaluation.evaluate(recommendation,truth)

print("Computing F-measure.")
from FMeasureEvaluation import FMeasureEvaluation
evaluation = FMeasureEvaluation()
evaluation.evaluate(recommendation,truth, 1)

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

print("Recall: {}, Precision: {}, MAP: {}".format(ev_recall,ev_precision,ev_map))