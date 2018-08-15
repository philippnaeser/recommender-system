# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:34:00 2018

@author: Steff
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###### Script parameters #######

TFIDF_MIN_DF = 0
TFIDF_MAX_DF = 1.0
TFIDF_NGRAM = (1,4)
TFIDF_MAX_FEATURES = 850000

MAX_RECS = 10

TRAINING_DATA_CONCAT = True
TRAINING_DATA = "small"
TEST_DATA = "small"

#CLASSIFIER = GaussianNB()
CLASSIFIER = MultinomialNB()
#CLASSIFIER = KNeighborsClassifier(20)
#CLASSIFIER = RandomForestClassifier()
#CLASSIFIER = SVC(probability=True)
#CLASSIFIER = AdaBoostClassifier(n_estimators=100)#,base_estimator=DecisionTreeClassifier(max_depth=8))

BATCHSIZE_EVALUATION = 200
PROCESSES_EVALUATION = 1

#################################

import sys
import os
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.getcwd(),".."))
sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
from TFIDFClassifierAbstractsModel import TFIDFClassifierAbstractsModel

# Generate model (main + child process).

model = TFIDFClassifierAbstractsModel(
            concat=TRAINING_DATA_CONCAT,
            classifier=CLASSIFIER,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM,
            max_features=TFIDF_MAX_FEATURES,
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
    import numpy as np
    import time

    # Train model if needed.
    
    if not model._has_persistent_model(TRAINING_DATA):
        d_train = DataLoader()
        d_train.training_data_for_abstracts(TRAINING_DATA)
        
        model.train(d_train.data,TRAINING_DATA)

    # Load test query and truth values.
    
    d_test = DataLoader()
    query_test, truth = d_test.evaluation_data_for_abstracts(TEST_DATA)
   
   # Apply test query and retrieve results.
    
    minibatches = np.array_split(query_test,int(len(query_test)/BATCHSIZE_EVALUATION))
    
    conferences = list()
    confidences = list()
    
    # Batchify the query to avoid OutOfMemory exceptions.
    
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
    
    model._load_model(TRAINING_DATA)
        
    for result in results:
        conferences.extend(result[0])
        confidences.extend(result[1])
    
    recommendation = [conferences,confidences]
    
    # Evaluate.
    
    from EvaluationContainer import EvaluationContainer
    evaluation = EvaluationContainer()
    evaluation.evaluate(recommendation,truth)
    
    print(CLASSIFIER)