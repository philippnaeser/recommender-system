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
sys.path.insert(0, os.path.join(os.getcwd(),"..","model_keywords_tfidf_union"))

from TfIdfUnionAbstractsModel import TfIdfUnionAbstractsModel
from CNNAbstractsModel import CNNAbstractsModel
from KeywordsUnionAbstractsModel import KeywordsUnionAbstractsModel
from EnsembleStackModel import EnsembleStackModel

# Load models.

model_tfidf = TfIdfUnionAbstractsModel(
        concat=True,
        min_df=0,
        max_df=1.0,
        ngram_range=(1,4),
        max_features=1000000,
        recs=10
)
model_tfidf._load_model(TRAINING_DATA)

model_cnn = CNNAbstractsModel("CNN2-100f-2fc-0.0005decay",recs=10)

model_keywords = KeywordsUnionAbstractsModel(
        concat=False,
        recs=10
)
model_keywords._load_model(TRAINING_DATA)

model = EnsembleStackModel(
            models=[
                    model_tfidf
                    ,model_cnn
                    #,model_keywords
            ],
            is_abstract=[
                    True
                    ,True
                    #,False
            ]
)

# Main script.

from DataLoader import DataLoader
import numpy as np
from TimerCounter import Timer

timer = Timer()

# Train ensemble.

if not model._has_persistent_model(TRAINING_DATA):
    d_train = DataLoader()
    d_train.training_data_for_abstracts_and_keywords(TRAINING_DATA)
    model.train(d_train.data,TRAINING_DATA)
    
"""
    
# Generate test query and truth values.

d_test = DataLoader()
query_test_abstract, query_test_keywords, truth = d_test.evaluation_data_for_abstracts_and_keywords(TEST_DATA)

# Apply test query and retrieve results.

minibatches_abstract = np.array_split(query_test_abstract,int(len(query_test_abstract)/BATCHSIZE_EVALUATION))
minibatches_keywords = np.array_split(query_test_keywords,int(len(query_test_keywords)/BATCHSIZE_EVALUATION))

conferences_rec = list()
confidences_rec = list()
#timer.set_counter(len(d_test.data.chapter_abstract))
timer.set_counter(len(minibatches_abstract))
#for abstract in d_test.data.chapter_abstract:
for i, abstracts in enumerate(minibatches_abstract):
    #confer, confid = model.query_single(abstract)
    #conferences_rec.append(confer)
    #confidences_rec.append(confid)
    
    confer, confid = model.query_batch(abstracts.tolist(),minibatches_keywords[i].tolist())
    conferences_rec.extend(confer)
    confidences_rec.extend(confid)
    timer.count()
    
recommendation = [conferences_rec,confidences_rec]

# Evaluate.

from EvaluationContainer import EvaluationContainer
evaluation = EvaluationContainer()
evaluation.evaluate(recommendation,truth)

print("#Recs: {}".format(len(recommendation[0])))
"""