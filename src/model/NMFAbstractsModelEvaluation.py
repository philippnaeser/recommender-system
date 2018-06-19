# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
@author: Andreea
"""


# query batchifier to avoid OutOfMemory exceptions
class Batchifier():
    def __init__(self):
        self.recv, self.send_c = Pipe()
        self.recv_c, self.send = Pipe()

    def run(self,model,batch):
        p = Process(target=self.f, args=(self.recv_c, self.send_c))
        p.start()
        
        self.send.send([model,batch])
        self.send.close()
        
        results = self.recv.recv()
        
        p.join()
        
        return results

    def f(self, recv, send):
        model, batch = recv.recv()
        results = model.query_batch(batch)
        
        send.send(results)
        send.close()

if __name__ == '__main__':
    import sys
    sys.path.insert(0, ".\..\data")
    
    from NMFAbstractsModel import NMFAbstractsModel
    from DataLoader import DataLoader
    import pandas as pd
    import numpy as np
    from multiprocessing import Process, Pipe
    
    ### load training data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.nmf.train.pkl"
    
    d_train = DataLoader()
    if not d_train.get_persistent(filename):
        d_train.papers(["2013","2014","2015"]).abstracts().conferences()
        d_train.data = d_train.data[["chapter","chapter_abstract","conference","conference_name"]].copy()
        d_train.data.drop(
            list(d_train.data[pd.isnull(d_train.data.chapter_abstract)].index),
            inplace=True
        )
        d_train.make_persistent(filename)
    
    model = NMFAbstractsModel()
    model.train(d_train.data)
    
    ### load test data if it is already pickled, otherwise create it from scratch
    
    filename = "abstracts.nmf.test.pkl"
    
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
        results = batchifier.run(model,minibatch)
        conferences.extend(results[0])
        confidences.extend(results[1])
        
    recommendation = [conferences,confidences]
        
    ### evaluate
    print("Computing MAP.")
    from MAPEvaluation import MAPEvaluation
    evaluation = MAPEvaluation()
    evaluation.evaluate(recommendation, truth)