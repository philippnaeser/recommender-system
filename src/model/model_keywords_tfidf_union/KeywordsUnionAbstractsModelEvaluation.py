# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
"""

###### Script parameters #######

MAX_RECS = 10

TRAINING_DATA = "small"
TRAINING_DATA_CONCAT = False
TEST_DATA = "small"

BATCHSIZE_EVALUATION = 200
PROCESSES_EVALUATION = 2

#################################

import os
import sys
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
from KeywordsUnionAbstractsModel import KeywordsUnionAbstractsModel

# Generate model (main + child process).

model = KeywordsUnionAbstractsModel(
        concat=TRAINING_DATA_CONCAT,
        recs=MAX_RECS
)

# Method to run in a multiprocessing process.

def evaluate_model(batch):
    result = model.query_batch(batch)
    return result

# Load model in child process.

if __name__ != '__main__':
    sys.stderr = open("debug-multiprocessing.err."+str(os.getppid())+".txt", "w")
    sys.stdout = open("debug-multiprocessing.out."+str(os.getppid())+".txt", "w")
    model._load_model(TRAINING_DATA)

# Main script.

if __name__ == '__main__':
    from DataLoader import DataLoader
    import numpy as np
    import time

    # Train model if needed.
    
    if not model._has_persistent_model(TRAINING_DATA):
        d_train = DataLoader()
        d_train.training_data_for_keywords(TRAINING_DATA)
        model.train(d_train.data,TRAINING_DATA)
        
    # Generate test query and truth values.
    
    d_test = DataLoader()
    query_test, truth = d_test.evaluation_data_for_keywords(TEST_DATA)

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
    
    print("After querying: {}".format(time.time()-now))
    
    # Evaluate.
    
    from EvaluationContainer import EvaluationContainer
    evaluation = EvaluationContainer()
    evaluation.evaluate(recommendation,truth)
    
    print("#Recs: {}, Feats: {}".format(len(recommendation[0]), model.stem_matrix.shape))