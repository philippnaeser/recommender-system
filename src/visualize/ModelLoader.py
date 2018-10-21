# -*- coding: utf-8 -*-
import sys
import os
import pickle
import pandas as pd

sys.path.insert(0, os.path.join(".","..","data"))
from DataLoader import DataLoader
sys.path.insert(0, os.path.join("..","model"))
sys.path.insert(0, os.path.join("..", "model", "model_authors_union"))
from BaselineModel import BaselineModel
sys.path.insert(0, os.path.join("..", "model", "model_tfidf_union"))
from TfIdfUnionAbstractsModel import TfIdfUnionAbstractsModel
sys.path.insert(0, os.path.join("..", "model", "model_doc2vec_union"))
from Doc2VecUnionAbstractsModel import Doc2VecUnionAbstractsModel
sys.path.insert(0, os.path.join("..", "model", "model_keywords_tfidf_union"))
from KeywordsUnionAbstractsModel import KeywordsUnionAbstractsModel
sys.path.insert(0, os.path.join("..", "model", "model_cnn2"))
from CNNAbstractsModel import CNNAbstractsModel
sys.path.insert(0, os.path.join("..", "model", "model_ensemble_stack"))
from EnsembleStackModel import EnsembleStackModel


class ModelLoader():

    def __init__(self):
        self.models = []
        print("preparing models")
        #d = DataLoader()
        #d.training_data("small").contributions()
        #self.data_authors = d.data[["author_name","conferenceseries_name"]].copy()
        #self.data_authors.columns = ["author_name","conferenceseries"]
        file = os.path.join(".", "data", "data_authors.pkl")
        #with open(file,"wb") as f:
        #    pickle.dump(self.data_authors, f)
        with open(file,"rb") as f:
                self.data_authors = pickle.load(f)
        #del d
        self.model_authors = BaselineModel()
        self.model_authors.train(self.data_authors)
        self.models.append("Authors")
        d = DataLoader()
        d.papers(["2013","2014","2015"]).conferences().conferenceseries().keywords()
        self.data_tags = d.data.loc[:, ["keyword", "keyword_label"]]
        #d = DataLoader()
        #d.training_data_for_abstracts("small")
        #self.data_abstracts = d.data.copy()
        file = os.path.join(".", "data", "data_abstracts.pkl")
        #with open(file,"wb") as f:
        #    pickle.dump(self.data_abstracts, f)
        with open(file,"rb") as f:
                self.data_abstracts = pickle.load(f)
        #del d
        self.model_tfidf_union = TfIdfUnionAbstractsModel(ngram_range=(1,4), max_features=1000000)
        self.model_tfidf_union.train(data=self.data_abstracts, data_name="small")
        self.models.append("nTfIdf_concat")
        self.model_doc2vec = Doc2VecUnionAbstractsModel(embedding_model="d2v_100d_w5_NS")
        self.model_doc2vec.train(data=self.data_abstracts, data_name="small")
        self.models.append("Doc2Vec")
        self.model_cnn = CNNAbstractsModel(net_name="CNN2-100f-2fc-0.0005decay")
        self.models.append("CNN")
        self.model_keyword = KeywordsUnionAbstractsModel()
        self.model_keyword._load_model("small")
        self.models.append("Keywords_TfIdf")
        ensemble_tfidf = TfIdfUnionAbstractsModel(ngram_range=(1,4), max_features=1000000, recs=100)
        ensemble_tfidf.train(data=self.data_abstracts, data_name="small")
        ensemble_cnn = CNNAbstractsModel(net_name="CNN2-100f-2fc-0.0005decay", recs=100)
        ensemble_keyword = KeywordsUnionAbstractsModel(recs=100)
        ensemble_keyword._load_model("small")
        self.model_ensemble = EnsembleStackModel(
            models=[
                    ensemble_tfidf
                    ,ensemble_cnn
                    ,ensemble_keyword
            ],
            is_abstract=[
                    True
                    ,True
                    ,False
            ],
            max_recs_models=100
        )
        self.model_ensemble._load_model("small")
        self.models.append("Ensemble")
        #We need to get series names, not scigraph links
        #d = DataLoader()
        #d.papers(["2013","2014","2015"]).conferences().conferenceseries()
        #self.data = d.data.copy()
        file = os.path.join(".", "data", "data.pkl")
        #with open(file,"wb") as f:
        #    pickle.dump(self.data, f)
        with open(file,"rb") as f:
                self.data = pickle.load(f)
        #Load the wikicfp dictionary
        file = os.path.join(".", "data", "WikiCFP_data.pkl")
        with open(file,"rb") as f:
                self.wikicfp = pickle.load(f)   
        print("Number of keys in wikicfp dictionary: ", len(self.wikicfp))
        print("Model Loader ready, models available:")
        print(self.models)

    def getModels(self):
        return self.models

    def query(self,modelName, data):
        print("querying model: " + modelName)
        if modelName=="Authors":
            rec = self.model_authors.query_single(data)
            return self.addDummyConfidence(rec)
        if modelName=="nTfIdf_concat":
            rec = self.model_tfidf_union.query_single(data)
            return self.getSeriesNames(rec)
        if modelName=="Doc2Vec":
            rec = self.model_doc2vec.query_batch(data)
            return self.getSeriesNames(rec)
        if modelName=="CNN":
            rec = self.model_cnn.query_single(data)
            return self.getSeriesNames(rec)
        if modelName=="Keywords_TfIdf":
            rec = self.model_keyword.query_single(self.getKeywordIDs(data))
            return self.getSeriesNames(rec)
        print("Model not found, please select a different model")
        return False
    
    def query_ensemble(self, abstract, keywords):
        print("querying ensemble model")
        keys = self.getKeywordIDs(keywords)
        rec = self.model_ensemble.query_single(abstract, keys)
        return self.getSeriesNames(rec)

    def autocomplete(self, modelName, data):
        if modelName == "Authors":
            return self.model_authors.get_author_names(term=data)
        if modelName == "Keywords_TfIdf" or modelName=="Ensemble":
            tags = pd.Series(self.data_tags["keyword_label"].unique())
        
            tags = tags[tags.str.lower().str.startswith(data.lower())][:10]
        
            return tags
    
    #Here we not only get the names, but also the additional info if available    
    def getSeriesNames(self, recommendation):
        #print(recommendation)
        conferenceseries = list()
        confidence = list()
        additional = list()
        for i,conf in enumerate(recommendation[0][0]):
            conferenceseries.append(self.data[self.data.conferenceseries==conf].iloc[0]["conferenceseries_name"])
            confidence.append(round(recommendation[1][0][i], 2))
            additional.append(self.addWikiCFP(conf))
        return [conferenceseries, confidence, additional]

    def addDummyConfidence(self, recommendation):
        conferenceseries = recommendation[0]
        confidence = list()
        additional = list()
        #Only for Author model, since it only responds with conferenceseries, no confidence
        for i in range(len (conferenceseries)):
            confidence.append(" ")
            additional.append(None)
        return [conferenceseries, confidence, additional]
    
    def addWikiCFP(self, conferenceseries):
        #Note: this is test only, delete after
        if conferenceseries in self.wikicfp:
            additional = self.wikicfp[conferenceseries]
            # Cut the description (did not work directly in a template)
            if additional['Description'] is not None:
                additional['Description'] = additional['Description'][:400]
            return additional
        else: 
            return None
        
    def getKeywordIDs(self, data):
        ids = ""
        for d in data:
            if d is not "":
                tmp = self.data_tags[self.data_tags.keyword_label == d].iloc[0].keyword
                tmp = tmp.replace("<http://scigraph.springernature.com/things/product-market-codes/","")
                tmp = tmp[0:-1]
                ids += tmp + " "        
        return ids