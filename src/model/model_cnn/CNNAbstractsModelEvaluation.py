# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:34:53 2018

@author: Steff
"""

MODEL_NAME = "CNN-100d-w2v-100f"
EMBEDDINGS_MODEL = "w2v_100d_w10_SG_NS"

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
from TimerCounter import Timer

timer = Timer()


### load test data

d_test = DataLoader()
d_test.test_data("small").abstracts()
d_test.data = d_test.data[["chapter_abstract","conferenceseries"]].copy()
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
        "test-data-cnn-smallw2v_100d_w10_SG_NS",
        "classes.pkl"
)

with open(classes_file,"rb") as f:
    classes = pickle.load(f)

net = CNNet(
        MODEL_NAME,
        embedding_size=100,
        classes=len(classes),
        filters=100
)
net.load_state()
net.cpu()

parser = EmbeddingsParser()
parser.load_model(EMBEDDINGS_MODEL)

model = CNNAbstractsModel()
model.train(
        net,
        parser,
        classes
)

query_test = list(d_test.data.chapter_abstract)#[0:1000]

conferences_truth = list()
confidences_truth = list()

for conference in list(d_test.data.conferenceseries):
    conferences_truth.append([conference])
    confidences_truth.append([1])
    
truth = [conferences_truth,confidences_truth]
    
### apply test query and retrieve results

recommendations_file = os.path.join(
        os.path.realpath(__file__),
        "..",
        "..",
        "..",
        "..",
        "data",
        "processed",
        "nn",
        MODEL_NAME,
        "evaluation.pkl"
        
)

try:
    with open(recommendations_file,"rb") as f:
        recommendations = pickle.load(f)
    print("loaded recommendations from disk.")
        
except FileNotFoundError:
    print("recommendations not found on disk, generating it.")
    conferences_rec = list()
    confidences_rec = list()
    timer.set_counter(len(d_test.data.chapter_abstract))
    for abstract in d_test.data.chapter_abstract:
        confer, confid = model.query_single(abstract)
        conferences_rec.append(confer)
        confidences_rec.append(confid)
        timer.count()
        
    recommendation = [conferences_rec,confidences_rec]
    
    with open(recommendations_file,"wb") as f:
        pickle.dump(recommendation,f)

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