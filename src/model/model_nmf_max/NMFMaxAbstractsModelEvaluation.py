# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 21:26:40 2018

@author: Steff
"""

###### Script parameters #######

""" NMF_INIT
"random": non-negative random matrices, scaled with: sqrt(X.mean() / n_components)
"nndsvd": Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness)
"nndsvda": NNDSVD with zeros filled with the average of X (better when sparsity is not desired)
"nndsvdar": NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired)
"""

NMF_TOPICS = 5
NMF_BETA_LOSS = "frobenius" # frobenius # kullback-leibler
NMF_SOLVER = "cd" # cd # mu
NMF_ALPHA = 1
NMF_RANDOM_STATE = 0
NMF_VERBOSE = True
NMF_INIT = "random"
NMF_MAX_ITER = 5

TFIDF_MIN_DF = 0
TFIDF_MAX_DF = 1.0

MAX_RECS = 10

TRAINING_DATA = "small"
TEST_DATA = "small"

BATCHSIZE_EVALUATION = 200
PROCESSES_EVALUATION = 2

#################################

import sys
import os
import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
from NMFMaxAbstractsModel import NMFMaxAbstractsModel

# Method to run in a multiprocessing process.

def evaluate_model(batch):
    result = model.query_batch(batch)
    return result

# Create model in main and child processes.
    
model = NMFMaxAbstractsModel(
        topics=NMF_TOPICS,
        beta_loss=NMF_BETA_LOSS,
        solver=NMF_SOLVER,
        alpha=NMF_ALPHA,
        random_state=NMF_RANDOM_STATE,
        verbose=NMF_VERBOSE,
        init=NMF_INIT,
        max_iter=NMF_MAX_ITER,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        recs=MAX_RECS
)

# Child process.

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
        d_train.training_data_for_abstracts(TRAINING_DATA)
        model.train(d_train.data,TRAINING_DATA)

    ### Load test query and truth values.  
    d_test = DataLoader()
    query_test, truth = d_test.evaluation_data_for_abstracts(TEST_DATA)
   
   # Apply test query and retrieve results.
    
    minibatches = np.array_split(query_test,int(len(query_test)/BATCHSIZE_EVALUATION))
    
    conferences = list()
    confidences = list()
    
    # Batchify the query to avoid OutOfMemory exceptions.
    
    ###################### MP VERSION POOL #######################

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
     
    ###################### SP VERSION ############################
    """
    model._load_model(TRAINING_DATA)

    for index, minibatch in enumerate(minibatches,1):
        print("Running minibatch [{}/{}]".format(index,len(minibatches)))
        results = model.query_batch(minibatch)
        conferences.extend(results[0])
        confidences.extend(results[1])
    """
    ##############################################################
    
    recommendation = [conferences,confidences]
    
    # Evaluate.
    
    from EvaluationContainer import EvaluationContainer
    evaluation = EvaluationContainer()
    evaluation.evaluate(recommendation,truth)