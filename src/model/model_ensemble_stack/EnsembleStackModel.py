# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import os
import pickle
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from TimerCounter import Timer

class EnsembleStackModel(AbstractModel):
    
    ##########################################
    def __init__(self,models,is_abstract,max_recs_models=10,recs=10):
        # number of recommendations to return
        self.recs = recs
        self.max_recs_models = max_recs_models
        self.models = models
        #self.sm = Softmax(dim=1)
        self.is_abstract = is_abstract
        
        #description = "-".join([
        #        str(concat),
        #        "{}"
        #])
        
        self.path = os.path.join(os.path.dirname(__file__), "..","..","..","data","processed","model_ensemble_stack")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file = os.path.join(
                self.path,
                "model.pkl"
        )
        
        self.persistent_vectors = os.path.join(
                self.path,
                "vectors.pkl"
        )
    
    ##########################################
    def query_single(self,abstract,keywords):
        """
            Queries the model and returns a list of recommendations.
            
            Args:
                keywords (str): The keywords as a concatenated string.
            
            Returns:
                str[]: name of the conference
                double[]: confidence scores
        """
        return self.query_batch([abstract],[keywords])
    
    ##########################################
    def query_batch(self,batch_abstract,batch_keywords):
        """
            Queries the model and returns a list of recommendations for each request.
            
            Args:
                batch[str]: The list of keywords as a concatenated strings.
            
            Returns:
                A list of size 'len(batch)' which contains the recommendations for each item of the batch.
                If author not found, the value is None.
                
                str[]: name of the conference
                double[]: confidence scores
        """
        vectors = np.zeros((len(batch_abstract),self.len_vec))
        row_len = len(batch_abstract)
        self.row_indices = np.repeat(np.arange(row_len),self.max_recs_models).reshape((row_len,self.max_recs_models))
        
        for i_m, m in enumerate(self.models):
            if self.is_abstract[i_m]:
                rec = m.query_batch(batch_abstract)
            else:
                rec = m.query_batch(batch_keywords)
                
            v = self._recs2vec(rec)
            vectors[:,i_m*self.len_truth:(i_m+1)*self.len_truth] = v
        
        predicts = self.classifier.predict_proba(vectors)
        o = np.argsort(-np.array(predicts))
        #print(o[:][0:self.recs])
        
        conferences = list()
        confidences = list()
        for index, order in enumerate(o):
            conferences.append(
                    self.truth[order[0:self.recs]]
            )
            confidences.append(
                    predicts[index][order][0:self.recs]
            )
            
            
        """
        for i, q in enumerate(recs[0]):
            for k in q.keys():
                for r in recs[q][1:]:
                    recs[q][0][k] *= r[k]
                    
        conferences = list()
        confidences = list()
        
        o = np.argsort(-np.array(sim))
        
        for index, order in enumerate(o):
            conferences.append(
                    list(self.data.iloc[order][0:self.recs].conferenceseries)
            )
            confidences.append(
                    list(sim[index][order][0:self.recs])
            )
        """


        return [conferences,confidences]
    
    ##########################################
    def _rec2dic(self,rec):
        dics = []
        for i, recs in enumerate(rec[0]):
            dic = {}
            for c, conf in enumerate(recs):
                dic[conf] = rec[1][i][c]
            dics.append(dic)
            
        return dics
    
    ##########################################
    def train(self,data,data_name):
        if not self._load_model(data_name):
            print("Model not persistent yet. Creating model.")

            self.truth = np.array(sorted(list(data.conferenceseries.unique())))
            self.len_truth = len(self.truth)
            self.len_vec = self.len_truth*len(self.models)
            
            # Generate new dataset for classifier.
            
            if os.path.isfile("train"):
                with open("train","rb") as f:
                    self.vectors = pickle.load(f)
                    print("Vectors loaded.")
            else:
                print("Generating vectors.")
                query_abstracts = list(data.chapter_abstract)
                query_keywords = list(data.keyword)
                
                minibatches_abstract = np.array_split(query_abstracts,int(len(query_abstracts)/200))
                minibatches_keywords = np.array_split(query_keywords,int(len(query_keywords)/200))
    
                self.vectors = np.zeros((len(data),self.len_vec))
                
                timer = Timer()
                
                row_len = 0
                for i_m, m in enumerate(self.models):
                    row = 0
                    timer.set_counter(len(data))
                    for i_b in range(len(minibatches_abstract)):
                        if self.is_abstract[i_m]:
                            if row_len != len(minibatches_abstract[i_b]):
                                row_len = len(minibatches_abstract[i_b])
                                self.row_indices = np.repeat(np.arange(row_len),self.max_recs_models).reshape((row_len,self.max_recs_models))
                            rec = m.query_batch(minibatches_abstract[i_b].tolist())
                        else:
                            if row_len != len(minibatches_keywords[i_b]):
                                row_len = len(minibatches_keywords[i_b])
                                self.row_indices = np.repeat(np.arange(row_len),self.max_recs_models).reshape((row_len,self.max_recs_models))
                            rec = m.query_batch(minibatches_keywords[i_b].tolist())
                        
                        v = self._recs2vec(rec)
                        self.vectors[row:(row+row_len),i_m*self.len_truth:(i_m+1)*self.len_truth] = v
                        row += row_len
                        timer.count(row_len)
                        
                with open("train","wb") as f:
                    pickle.dump(self.vectors, f)
                    
            print("Training classifier.")
            self.classifier = LogisticRegression()#(verbose=1)
            self.classifier.fit(self.vectors,data.conferenceseries)
                    
    ##########################################
    def _recs2vec(self,rec):
        vec = np.zeros((len(rec[0]),self.len_truth))
        
        indices = self.truth.searchsorted(rec[0])
        vec[self.row_indices,indices] = rec[1]
            
        return vec
            
    ##########################################
    def _file(self,data_name):
        return self.persistent_file.format(data_name)
    
    ##########################################
    def _save_model(self,data_name):
        file = self._file(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.truth, self.len_truth, self.len_vec, self.vectors, self.classifier, self.max_recs_models], f)
    
    ##########################################
    def _load_model(self,data_name):
        file = self._file(data_name)
        if os.path.isfile(file):
            with open(file,"rb") as f:
                print("Loading persistent model.")
                self.truth, self.len_truth, self.len_vec, self.vectors, self.classifier, self.max_recs_models = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
    
    ##########################################
    def _has_persistent_model(self,data_name):
        return os.path.isfile(self._file(data_name))