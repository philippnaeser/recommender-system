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

# Load training data and train model.

d = DataLoader()
d.training_data("small").contributions()

model = BaselineModel()
model.train(d.data)

# Generate test data.

d = DataLoader()
d.test_data("small").contributions()
d.data.author_name = d.data.author_name.str.decode("unicode_escape").str.lower()

conferenceseries = d.data[["chapter","conferenceseries"]].copy().drop_duplicates().reset_index()
authors = d.data.groupby("chapter")["author_name"].apply(list)

data_test = conferenceseries.join(
        authors,
        on="chapter"
)

query = list(data_test["author_name"])
truth = [list(data_test["conferenceseries"].apply(lambda x:[x]))]

print("Getting recommendations.")
recommendation = model.query_batch(query)

# Evaluation.

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

num_none = sum(x is None for x in recommendation[0])
size_query = len(query)
print("{}/{} recommendations None = {}%.".format(num_none,size_query,num_none/size_query*100))