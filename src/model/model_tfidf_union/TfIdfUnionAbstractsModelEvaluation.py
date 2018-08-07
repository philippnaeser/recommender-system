# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
"""

###### Script parameters #######

TFIDF_MIN_DF = 0
TFIDF_MAX_DF = 1.0

MAX_RECS = 10

TRAINING_DATA = "small"
TEST_DATA = "small"

BATCHSIZE_EVALUATION = 200
PROCESSES_EVALUATION = 8

#################################

import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
from TfIdfAbstractsModel import TfIdfUnionAbstractsModel

# Generate model (main + child process).

model = TfIdfUnionAbstractsModel(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        recs=MAX_RECS
)

# Method to run in a multiprocessing process.

def evaluate_model(batch):
    result = model.query_batch(batch)
    return result

# Load model in child process.

if __name__ != '__main__':
    
    #sys.stderr = open("debug-multiprocessing.err."+str(os.getppid())+".txt", "w")
    #sys.stdout = open("debug-multiprocessing.out."+str(os.getppid())+".txt", "w")

    model._load_model(TRAINING_DATA)

# Main script.

if __name__ == '__main__':
    from DataLoader import DataLoader
    import pandas as pd
    import numpy as np
    import time

    # Train model if needed.
    
    if not model._has_persistent_model(TRAINING_DATA):
        d_train = DataLoader()
        d_train.training_data(TRAINING_DATA).abstracts()
        d_train.data = d_train.data[["chapter_abstract","conferenceseries"]].copy()
        d_train.data.drop(
            list(d_train.data[pd.isnull(d_train.data.chapter_abstract)].index),
            inplace=True
        )
        d_train.data.chapter_abstract = d_train.data.chapter_abstract.str.decode("unicode_escape")
        
        model.train(d_train.data,TRAINING_DATA)
        
    # Generate test data.
    
    d_test = DataLoader()
    d_test.test_data(TEST_DATA).abstracts()
    d_test.data = d_test.data[["chapter_abstract","conferenceseries"]].copy()
    d_test.data.drop(
        list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
        inplace=True
    )
    d_test.data.chapter_abstract = d_test.data.chapter_abstract.str.decode("unicode_escape")

    # Generate test query and truth values.

    query_test = list(d_test.data.chapter_abstract)#[0:1000]
    
    conferences_truth = list()
    confidences_truth = list()
    
    for conference in list(d_test.data.conferenceseries):
        conferences_truth.append([conference])
        confidences_truth.append([1])
        
    truth = [conferences_truth,confidences_truth]
        
    # Apply test query and retrieve results.
    
    minibatches = np.array_split(query_test,int(len(query_test)/BATCHSIZE_EVALUATION))
    
    conferences = list()
    confidences = list()
    
    # Batchify the query to avoid OutOfMemory exceptions.
    
    now = time.time()

    results = None
    
    def process_ready(r):
        global results
        results = r
    
    pool = mp.Pool(processes=PROCESSES_EVALUATION)
    job = pool.map_async(evaluate_model,minibatches,callback=process_ready)    
    pool.close()
    
    while (True):
        if (job.ready()): break
        print("Tasks remaining: {}".format(job._number_left*job._chunksize))
        time.sleep(5)
        
    print("Tasks completed.")
            
    for result in results:
        conferences.extend(result[0])
        confidences.extend(result[1])
        
    model._load_model(TRAINING_DATA)
        
    recommendation = [conferences,confidences]
    
    print("After querying: {}".format(now-time.time()))
    
    # Evaluate.
    
    from EvaluationContainer import EvaluationContainer
    evaluation = EvaluationContainer()
    evaluation.evaluate(recommendation,truth)
    
    print("#Recs: {}, Feats: {}".format(len(recommendation[0]), model.stem_matrix.shape))