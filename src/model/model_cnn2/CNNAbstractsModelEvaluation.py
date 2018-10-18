# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:34:53 2018

@author: Steff
"""

MODEL_NAME = "CNN2-100f-2fc-0.0005decay"
MODEL_EPOCH = 150

import sys
import os
import pickle

sys.path.insert(0, os.path.join(
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
from CNNAbstractsModel import CNNAbstractsModel
from TimerCounter import Timer

timer = Timer()

### Load test query and truth values.

d_test = DataLoader()
query_test, truth = d_test.evaluation_data_for_abstracts("small")

### Load model.

model = CNNAbstractsModel(
        net_name=MODEL_NAME,
        epoch=MODEL_EPOCH
)
    
### Apply test query and retrieve results.

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
        conferences_rec.extend(confer)
        confidences_rec.extend(confid)
        timer.count()
        
    recommendation = [conferences_rec,confidences_rec]
    
    with open(recommendations_file,"wb") as f:
        pickle.dump(recommendation,f)

### Evaluate.

from EvaluationContainer import EvaluationContainer
evaluation = EvaluationContainer()
evaluation.evaluate(recommendation, truth)