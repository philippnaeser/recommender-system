# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:34:53 2018

@author: Steff
"""

import sys
import os
import pickle

import pandas as pd

sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..")
)
print(os.path.join(
        os.path.realpath(__file__),"..","..")
)
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","neuralnets")
)
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","..","data")
)
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","evaluations")
)

from DataLoader import DataLoader
from CNNet import CNNet
from CNNAbstractsModel import CNNAbstractsModel
from EmbeddingsParser import EmbeddingsParser



### load test data if it is already pickled, otherwise create it from scratch

d_test = DataLoader()
d_test.papers(["2016"]).abstracts().conferences().conferenceseries()
#d_test.data = d_test.data[["chapter","chapter_abstract","conference","conference_name"]].copy()
d_test.data.drop(
    list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
    inplace=True
)

### create test query and truth values

classes_file = os.path.join(
        os.path.realpath(__file__),
        "..",
        "..",
        "..",
        "..",
        "data",
        "interim",
        "neuralnet_training",
        "training-data-cnn-small6d50",
        "classes.pkl"
)

with open(classes_file,"rb") as f:
    classes = pickle.load(f)

net = CNNet(
        "CNN-CPU-test",
        embedding_size=50,
        classes=len(classes),
        filters=50
)
net.load_state()
net.cpu()

parser = EmbeddingsParser()
parser.load_model("6d50")

model = CNNAbstractsModel()
model.train(
        net,
        parser,
        classes
)

query_test = list(d_test.data.chapter_abstract)[0:1000]

conferences_truth = list()
confidences_truth = list()

for conference in list(d_test.data.conferenceseries):
    conferences_truth.append([conference])
    confidences_truth.append([1])
    
truth = [conferences_truth,confidences_truth]
    
### apply test query and retrieve results

recommendation = model.query_batch(query_test)
    
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

print("Recall: {}, Precision: {}, F1Measure: {}, MAP: {}".format(ev_recall, ev_precision, ev_fmeasure, ev_map))