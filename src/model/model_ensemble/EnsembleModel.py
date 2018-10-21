# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import numpy as np
from torch.nn import Softmax
import torch

class EnsembleModel(AbstractModel):
    
    ##########################################
    def __init__(self,models,is_abstract,apply_softmax,model_weight,recs=10):
        # number of recommendations to return
        self.recs = recs
        self.models = models
        self.sm = Softmax(dim=1)
        self.apply_softmax = apply_softmax
        self.is_abstract = is_abstract
        self.model_weight = model_weight
        
        """
        description = "-".join([
                str(concat),
                "{}"
        ])
        
        self.path = os.path.join("..","..","..","data","processed","model_ensemble")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file = os.path.join(
                self.path,
                "model-"+description+".pkl"
        )
        """
    
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
        conf = None
        scores_combined = None
        for i, m in enumerate(self.models):
            #recs.append(self._rec2dic(m.query_batch(batch)))
            if self.is_abstract[i]:
                results = m.query_batch(batch_abstract)
                batch_len = len(batch_abstract)
            else:
                results = m.query_batch(batch_keywords)
                batch_len = len(batch_keywords)
            conferences = np.array(results[0],dtype=str)
            if self.apply_softmax[i]:
                scores = torch.tensor(results[1])
                scores = self.sm(scores).numpy()
                #scores = np.array(results[1],dtype=np.float64)
            else:
                scores = np.array(results[1],dtype=np.float64)
            order = np.argsort(conferences)[0:batch_len,0:743]
            #print(order.shape)
            #print(conferences.shape)
            
            if conf is None:
                conf = conferences[0][order[0]]
            
            if scores_combined is None:
                scores_combined = scores[np.arange(np.shape(scores)[0])[:,np.newaxis], order] * self.model_weight[i]
            else:
                scores_combined += scores[np.arange(np.shape(scores)[0])[:,np.newaxis], order] * self.model_weight[i]
            
        conferences = list()
        confidences = list()
        
        o = np.argsort(-scores_combined)
        
        for index, order in enumerate(o):
            conferences.append(
                    list(conf[order][0:self.recs])
            )
            confidences.append(
                    list(scores_combined[index][order][0:self.recs])
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
    def train(self,data,data_name,topics_single,topics_multiple,topics_parents,topics_labels):
        if not self._load_model(data_name):
            print("Model not persistent yet. Creating model.")
            for check in ["chapter_abstract","conferenceseries"]:
                if not check in data.columns:
                    raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
            
            # concat abstracts if parameter set.
            if self.concat:
                data = data[["chapter_abstract","conferenceseries"]]
                data.chapter_abstract = data.chapter_abstract + " "
                data = data.groupby("conferenceseries").sum().reset_index()
            self.data = data
            
            # set object attributes for topic extraction.
            self.topics_single = set(topics_single)
            self.topics_multiple = topics_multiple
            self.topics_labels = topics_labels
            self.topics_parents = topics_parents
            
            # extract topics from training data.
            topics_sets = []
            topics_all = set()
            #now = time.time()
            for i, abstract in enumerate(data.chapter_abstract.str.lower()):
                topics = self.extract_topics(abstract)
                topics_sets.append(topics)
                topics_all.update(topics)
            
            # build topic matrix for training data.
            self.topics_all = list(topics_all)
            self.topics_matrix = self.indicator_matrix(topics_sets)
            
            # save model to disk.
            self._save_model(data_name)