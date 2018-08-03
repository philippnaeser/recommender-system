# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

import pandas as pd
import sys
import os
import pickle
print(os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(".","..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
from DataLoader import DataLoader
from TagModel import TagModel

"""
    Prepare the data for the evaluation.
"""
#Training part
d = DataLoader()
d.papers(["2013","2014","2015"]).conferences().conferenceseries().keywords()

data_train = d.data.loc[:, ["conferenceseries", "keyword_label"]]
data_train.columns = ["conferenceseries", "tag_name"]

model = TagModel()
model.train(data_train)

#Test Part
print("Preparing test data")
try:
    conferenceseries = pickle.load(open("conferenceseries.pkl", "rb"))
    tags = pickle.load(open("tags.pkl", "rb"))
except Exception as e:
    print("No saved testdata found, generating it will take a moment")
    d = DataLoader()
    d.papers(["2016"]).conferences().conferenceseries().keywords()
    temp = d.data.loc[:, ["chapter","conferenceseries", "keyword_label"]]
    temp.columns = ["chapter", "conferenceseries", "tag_name"]
    chapters = list(temp.chapter.unique())
    print(len(chapters))
    conferenceseries = list()
    tags = list()
    i = 0
    for chapter in chapters:
        conferenceseries.append(temp[temp["chapter"]==chapter].iloc[0].conferenceseries)
        tags.append(list(temp[temp["chapter"]==chapter].tag_name.unique()))
        i += 1
        if i%1000 == 0:
            print(i)
        if i == 21399:
            print(len(conferenceseries))
            print(len(tags))
    print("done")
    print(len(conferenceseries))
    print(len(tags))
    print("Saving data to pickle files")
    pickle.dump(conferenceseries, open("conferenceseries.pkl", "wb"))
    pickle.dump(tags, open("tags.pkl", "wb"))


print("Getting recommendations.")
recommendation = model.query_batch(tags)
print("Getting truth values.")
conferences_truth = list()
confidences_truth = list()
for conference in conferenceseries:
    conferences_truth.append([conference])
    confidences_truth.append([1])
truth = [conferences_truth,confidences_truth]
    

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