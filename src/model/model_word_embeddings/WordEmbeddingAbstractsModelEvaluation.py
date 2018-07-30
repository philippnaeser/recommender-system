# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:41:02 2018

@author: Andreea
"""

import os
import sys

# query batchifier to avoid OutOfMemory exceptions
class Batchifier():
    def __init__(self):
        self.recv, self.send_c = Pipe()
        self.recv_c, self.send = Pipe()

    def run(self,batch):
        return model.query_batch(batch)
        
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
        
        from WordEmbeddingAbstractsModel import WordEmbeddingAbstractsModel
        model = WordEmbeddingAbstractsModel()
        model._load_model()
        
        batch = recv.recv()
        results = model.query_batch(batch)
        
        send.send(results)
        send.close()

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.getcwd()))
    sys.path.insert(0, os.path.join(os.getcwd(),".."))
    sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
    sys.path.insert(0, os.path.join(os.getcwd(),"..","evaluations"))
    
    from WordEmbeddingAbstractsModel import WordEmbeddingAbstractsModel
    from DataLoader import DataLoader
    import pandas as pd
    import numpy as np
    from multiprocessing import Process, Pipe
    
    ### load training data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.wordembedding.train.pkl"
    
    d_train = DataLoader()
    if not d_train.get_persistent(filename):
        d_train.papers(["2013","2014","2015"]).abstracts().conferences().conferenceseries()
        d_train.data = d_train.data[["chapter","chapter_abstract","conference","conference_name", "conferenceseries"]].copy()
        d_train.data.drop(
            list(d_train.data[pd.isnull(d_train.data.chapter_abstract)].index),
            inplace=True
        )
        d_train.make_persistent(filename)
    
    model = WordEmbeddingAbstractsModel(pretrained = True)
    
    #One of the pretrained {"6d50","6d100","6d200","6d300","42d300","840d300", "word2vec", "fasttext"}.
    #One of the models trained on the abstracts' text data.
    embedding_model = "w2v_100d_w10_SG_NS"  
    
    model.train(d_train.data, embedding_model)
    
    ### load test data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.wordembedding.test.pkl"
    
    d_test = DataLoader()
    if not d_test.get_persistent(filename):
        d_test.papers(["2016"]).abstracts().conferences().conferenceseries()
        d_test.data = d_test.data[["chapter","chapter_abstract","conference","conference_name", "conferenceseries"]].copy()
        d_test.data.drop(
            list(d_test.data[pd.isnull(d_test.data.chapter_abstract)].index),
            inplace=True
        )
        d_test.make_persistent(filename)

    ### create test query and truth values
    query_test = list(d_test.data.chapter_abstract.str.decode("unicode_escape"))#[0:1000]
    
    conferences_truth = list()
    confidences_truth = list()
    
    for conference in list(d_test.data.conferenceseries):
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
        #results = model.query_batch(minibatch)
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
