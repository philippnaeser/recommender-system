# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
"""

###### Script parameters #######

MAX_RECS = 10

TRAINING_DATA = "small"
TRAINING_DATA_CONCAT = True
TEST_DATA = "small"

BATCHSIZE_EVALUATION = 200
PROCESSES_EVALUATION = 1

#################################

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","model_keywords_tfidf_union"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","model_tfidf_union"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","model_cnn2"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","neuralnets"))

from TfIdfUnionAbstractsModel import TfIdfUnionAbstractsModel
from CNNAbstractsModel import CNNAbstractsModel
from EnsembleModel import EnsembleModel

# Load models.

model_tfidf = TfIdfUnionAbstractsModel(
        concat=True,
        min_df=0,
        max_df=1.0,
        ngram_range=(1,4),
        max_features=1000000,
        recs=MAX_RECS
)
model_tfidf._load_model(TRAINING_DATA)

model_cnn = CNNAbstractsModel("CNN2-100f-2fc-0.0005decay",recs=MAX_RECS)

model = EnsembleModel([model_tfidf,model_cnn])

# Main script.

from DataLoader import DataLoader
import numpy as np
from TimerCounter import Timer

timer = Timer()
    
# Generate test query and truth values.

d_test = DataLoader()
query_test, truth = d_test.evaluation_data_for_abstracts(TEST_DATA)

# Apply test query and retrieve results.

minibatches = np.array_split(query_test,int(len(query_test)/BATCHSIZE_EVALUATION))

conferences_rec = list()
confidences_rec = list()
#timer.set_counter(len(d_test.data.chapter_abstract))
timer.set_counter(len(minibatches))
#for abstract in d_test.data.chapter_abstract:
for abstracts in minibatches:
    #confer, confid = model.query_single(abstract)
    #conferences_rec.append(confer)
    #confidences_rec.append(confid)
    
    confer, confid = model.query_batch(abstracts.tolist())
    conferences_rec.extend(confer)
    confidences_rec.extend(confid)
    timer.count()
    
recommendation = [conferences_rec,confidences_rec]

# Evaluate.

from EvaluationContainer import EvaluationContainer
evaluation = EvaluationContainer()
evaluation.evaluate(recommendation,truth)

print("#Recs: {}, Feats: {}".format(len(recommendation[0]), model.topics_matrix.shape))