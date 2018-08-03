# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 01:39:59 2018

@author: Philipp
"""

import os
import sys

indices = [10580,14601,21386,321]

sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),".."))
from DataLoader import DataLoader
from TagModel import TagModel

d = DataLoader()
d.papers(["2013","2014","2015"]).conferences().conferenceseries().keywords()

data_train = d.data.loc[:, ["conferenceseries", "keyword_label"]]
data_train.columns = ["conferenceseries", "tag_name"]

model = TagModel()
model.train(data_train)

d_test = DataLoader()
d_test.papers(["2016"]).conferences().conferenceseries().keywords()


for index_current in indices:
    data = d_test.data.iloc[[index_current]].copy()
    
    print("############################################")
    print("Querying abstract: {}".format(index_current))
    
    query_test = list(data.keyword_label.unique())
    
    conferences_truth = list()
    confidences_truth = list()
    
    for conference in list(data.conferenceseries):
        conferences_truth.append([conference])
        confidences_truth.append([1])
        
    truth = [conferences_truth,confidences_truth]
        
    ### apply test query and retrieve results
    
    conferences = list()
    confidences = list()
    
    results = model.query_batch(query_test)
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
    
    print("Recall: {}, Precision: {}, F1Measure: {}, MAP: {}".format(ev_recall, ev_precision, ev_fmeasure, ev_map))
    
    ##### delete after me
    
    print("###### RECOMMENDATIONS ######")
    for i,conf in enumerate(recommendation[0][0]):
        print("### "+str(i))
        print(d.data[d.data.conferenceseries==conf].iloc[0]["conferenceseries"])
        print(d.data[d.data.conferenceseries==conf].iloc[0]["conferenceseries_name"])
        print(recommendation[1][0][i])