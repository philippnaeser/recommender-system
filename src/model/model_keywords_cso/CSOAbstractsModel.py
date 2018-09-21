# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:37:51 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

class CSOAbstractsModel(AbstractModel):
    
    ##########################################
    def __init__(self,concat=True,recs=10):
        # number of recommendations to return
        self.recs = recs
        self.concat = concat
        
        description = "-".join([
                str(concat),
                "{}"
        ])
        
        self.path = os.path.join("..","..","..","data","processed","model_cso_union")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        self.persistent_file = os.path.join(
                self.path,
                "model-"+description+".pkl"
        )
    
    ##########################################
    def query_single(self,keywords):
        """
            Queries the model and returns a list of recommendations.
            
            Args:
                keywords (str): The keywords as a concatenated string.
            
            Returns:
                str[]: name of the conference
                double[]: confidence scores
        """
        return self.query_batch([keywords])
    
    ##########################################
    def query_batch(self,batch):
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
        conferences = list()
        confidences = list()
        #self.count_init(len(batch))
        
        topics = self.extract_topics_from_batch(batch)
        print("Topics extracted.")
        q_v = (self.indicator_matrix(topics))
        print("Topics transformed.")
        print("Dimensionality of batch: {}".format(q_v.shape))
        sim = cosine_similarity(q_v,self.topics_matrix)
        print("Cosine similarity computed.")
        o = np.argsort(-np.array(sim))
        index = 0
        
        for index, order in enumerate(o):
            data_conf = np.array(self.data["conferenceseries"])[order]
            data_sim = np.array(sim[index])[order]
            
            conference = list()
            confidence = list()
            i = 0
            while len(conference) < self.recs:
                c = data_conf[i]
                if c not in conference:
                    conference.append(c)
                    confidence.append(data_sim[i])   
                i += 1
                
            conferences.append(
                    conference
            )
            confidences.append(
                    confidence
            )
            #self.count()
            
        return [conferences,confidences]
    
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
    
    ##########################################
    def extract_topics_from_batch(self,batch):
        topics = []
        
        for a in batch:
            topics.append(self.extract_topics(a))
            
        return topics
    
    ##########################################
    def extract_topics(self,abstract):
        a = set(word_tokenize(abstract))
        topics = a.intersection(self.topics_single)
        for m in self.topics_multiple:
            if m in abstract:
                topics.add(m)
        
        # build up phase: get all the parents
        size = 0
        while size < len(topics):
            size = len(topics)
            for t in topics.copy():
                try:
                    topics.update(self.topics_parents[t])
                except KeyError:
                    pass
            
        # clean-up: remove all redundant elements
        for t in topics.copy():
            try:
                label = self.topics_labels[t]
                topics.discard(t)
                topics.add(label)
            except KeyError:
                pass
            
        return topics
            
    ##########################################
    def indicator_matrix(self,batch):
        matrix = np.zeros((len(batch),len(self.topics_all)),dtype=np.uint8)
        
        for i, t in enumerate(batch):
            matrix[i,:] = np.isin(self.topics_all,list(t))
            
        return matrix
        
    ##########################################
    def __call__(self, doc):
        # tokenize the input with a regex and stem each token
        return [self.stemmer.stem(t) for t in self.token_pattern.findall(doc)]
    
    ##########################################
    def _get_word_freq(self, matrix, vectorizer):
        '''Function for generating a list of (freq, word)'''
        return sorted([(matrix.getcol(idx).sum(), word) for word, idx in vectorizer.vocabulary_.items()], reverse=True)
    
    ##########################################
    def _file(self,data_name):
        return self.persistent_file.format(data_name)
    
    ##########################################
    def _save_model(self,data_name):
        file = self._file(data_name)
        with open(file,"wb") as f:
            pickle.dump([self.topics_matrix, self.topics_single, self.topics_multiple, self.topics_parents, self.topics_labels, self.topics_all, self.data], f)
    
    ##########################################
    def _load_model(self,data_name):
        file = self._file(data_name)
        if os.path.isfile(file):
            with open(file,"rb") as f:
                print("Loading persistent model.")
                self.topics_matrix, self.topics_single, self.topics_multiple, self.topics_parents, self.topics_labels, self.topics_all, self.data = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
    
    ##########################################
    def _has_persistent_model(self,data_name):
        return os.path.isfile(self._file(data_name))
    
    ##########################################
    def print_top_k(self, k):
        for tfidf, word in self._get_word_freq(self.stem_matrix, self.stem_vectorizer)[:k]:
            print("{:.3f} {}".format(tfidf, word))
