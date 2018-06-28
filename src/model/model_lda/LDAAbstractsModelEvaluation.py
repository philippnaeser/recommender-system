# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:18:11 2018

@author: Andreea
"""

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))


# query batchifier to avoid OutOfMemory exceptions
class Batchifier():
    def __init__(self):
        self.recv, self.send_c = Pipe()
        self.recv_c, self.send = Pipe()

    def run(self,batch):
        # return model.query_batch(batch)

        p = Process(target=self.f, args=(self.recv_c, self.send_c))
        p.start()
        
        self.send.send(batch)
        self.send.close()
        
        results = self.recv.recv()
        
        p.join()
        
        return results

    def f(self, recv, send):
        sys.stderr = open("debug-multiprocessing.err.txt", "w")
        sys.stdout = open("debug-multiprocessing.out.txt", "w")
        sys.path.insert(0, os.path.join(os.getcwd(),".."))
        sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
        sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
        
        from LDAAbstractsModel import LDAAbstractsModel
        model = LDAAbstractsModel()
        model._load_model_x()
        model._load_model_factors()
        
        batch = recv.recv()
        results = model.query_batch(batch)
        
        send.send(results)
        send.close()

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.getcwd()))
    sys.path.insert(0, os.path.join(os.getcwd(),".."))
    sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
    sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
    
    from LDAAbstractsModel import LDAAbstractsModel
    from DataLoader import DataLoader
    import pandas as pd
    import numpy as np
    from multiprocessing import Process, Pipe
    
    ### load training data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.lda.train.pkl"
    
    d_train = DataLoader()
    if not d_train.get_persistent(filename):
        d_train.papers(["2013","2014","2015"]).abstracts().conferences()
        d_train.data = d_train.data[["chapter","chapter_abstract","conference","conference_name"]].copy()
        d_train.data.drop(
            list(d_train.data[pd.isnull(d_train.data.chapter_abstract)].index),
            inplace=True
        )
        d_train.make_persistent(filename)
    
    model = LDAAbstractsModel()
    dimensions = 500
    model.train(d_train.data, dimensions)
     
    ### load test data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.lda.test.pkl"
    
    d_test = DataLoader()
    if not d_test.get_persistent(filename):
        d_test.papers(["2016"]).abstracts().conferences()
        d_test.data = d_test.data[["chapter","chapter_abstract","conference","conference_name"]].copy()
        d_test.data.drop(
            list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
            inplace=True
        )
        d_test.make_persistent(filename)

    ### create test query and truth values

    query_test = list(d_test.data.chapter_abstract)#[0:1000]
    
    conferences_truth = list()
    confidences_truth = list()
    
    for conference in list(d_test.data.conference_name):
        conferences_truth.append([conference])
        confidences_truth.append([1])
        
    truth = [conferences_truth,confidences_truth]
        
    ### apply test query and retrieve results
    
    batchsize = 200
    minibatches = np.arange(0,len(query_test),batchsize)
    
    conferences = list()
    confidences = list()
    
    # batchify the query to avoid OutOfMemory exceptions
    for i in minibatches:
        minibatch = query_test[i:(i+batchsize)]
        batchifier = Batchifier()
        print("Running minibatch [{}/{}]".format(int((i/batchsize)+1),len(minibatches)))
        results = batchifier.run(minibatch)
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