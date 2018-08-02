# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 19:06:21 2018

@author: Steff
"""

import os
import sys

indices = [10580,14601,21386,321]
use_model = "CNN"

sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(),"model_tfidf"))
sys.path.insert(0, os.path.join(os.getcwd(),"model_nmf"))
sys.path.insert(0, os.path.join(os.getcwd(),"model_cnn"))
sys.path.insert(0, os.path.join(os.getcwd(),"neuralnets"))

from TfIdfAbstractsModel import TfIdfAbstractsModel
from NMFAbstractsModel import NMFAbstractsModel
from CNNAbstractsModel import CNNAbstractsModel
from CNNet import CNNet
from EmbeddingsParser import EmbeddingsParser
from DataLoader import DataLoader
import pandas as pd
import numpy as np
import pickle

d_test = DataLoader()
d_test.test_data("small").abstracts()
d_test.data = d_test.data[["chapter_abstract","conferenceseries"]].copy()
#d_test.data.drop(
#    list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
#    inplace=True
#)
d_test.data.chapter_abstract = d_test.data.chapter_abstract.str.decode("unicode_escape")

d = DataLoader()
d.conferenceseries()

#model = TfIdfAbstractsModel()
#model._load_model()

if use_model == "NMF":
    model = NMFAbstractsModel()
    model._load_model_lr()
    model._load_model_x()
elif use_model == "CNN":
    classes_file = os.path.join(
            os.path.realpath(__file__),
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
    
    MODEL_NAME = "CNN-100d-w2v-100f"
    EMBEDDINGS_MODEL = "w2v_100d_w10_SG_NS"
    
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

for index_current in indices:
    data = d_test.data.iloc[[index_current]].copy()
    
    print("############################################")
    print("Querying abstract: {}".format(index_current))
    
    query_test = list(data.chapter_abstract)
    
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