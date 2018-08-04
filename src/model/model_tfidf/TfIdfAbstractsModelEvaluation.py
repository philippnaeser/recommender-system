# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))

from TfIdfAbstractsModel import TfIdfAbstractsModel
from DataLoader import DataLoader
import pandas as pd
import numpy as np

### Generate training data and train model.

d_train = DataLoader()
d_train.training_data("small").abstracts()
d_train.data = d_train.data[["chapter_abstract","conferenceseries"]].copy()
d_train.data.drop(
    list(d_train.data[pd.isnull(d_train.data.chapter_abstract)].index),
    inplace=True
)
d_train.data.chapter_abstract = d_train.data.chapter_abstract.str.decode("unicode_escape")

model = TfIdfAbstractsModel()
model.train(d_train.data)

### load test data if it is already pickled, otherwise create it from scratch

d_test = DataLoader()
d_test.test_data("small").abstracts()
d_test.data = d_test.data[["chapter_abstract","conferenceseries"]].copy()
d_test.data.drop(
    list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
    inplace=True
)
d_test.data.chapter_abstract = d_test.data.chapter_abstract.str.decode("unicode_escape")

### create test query and truth values

query_test = list(d_test.data.chapter_abstract)#[0:1000]

conferences_truth = list()
confidences_truth = list()

for conference in list(d_test.data.conferenceseries):
    conferences_truth.append([conference])
    confidences_truth.append([1])
    
truth = [conferences_truth,confidences_truth]
    
### apply test query and retrieve results

batchsize = 200
minibatches = np.arange(0,len(query_test),batchsize)

conferences = list()
confidences = list()

# batchify the query to avoid OutOfMemory exceptions
for i in minibatches:
    minibatch = query_test[i:(i+batchsize)]
    print("Running minibatch [{}/{}]".format(int((i/batchsize)+1),len(minibatches)))
    results = model.query_batch(minibatch)
    conferences.extend(results[0])
    confidences.extend(results[1])
    
recommendation = [conferences,confidences]
    
### evaluate

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

print("Recall: {}, Precision: {}, F1Measure: {}, MAP: {}, #Recs: {}, Feats: {}".format(ev_recall, ev_precision, ev_fmeasure, ev_map, len(recommendation[0]), model.stem_matrix.shape))