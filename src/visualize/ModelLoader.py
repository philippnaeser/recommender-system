# -*- coding: utf-8 -*-
import sys
import os
import pickle

#sys.path.insert(0, os.path.join(".","..","data"))
#from DataLoader import DataLoader
sys.path.insert(0, os.path.join(".","models"))
from BaselineModel import BaselineModel
from TagModel import TagModel
from TfIdfMaxAbstractsModel import TfIdfMaxAbstractsModel
from TfIdfUnionAbstractsModel import TfIdfUnionAbstractsModel
#from TFIDFClassifierAbstractsModel import TFIDFClassifierAbstractsModel
#from sklearn.naive_bayes import MultinomialNB
from NMFAbstractsModel import NMFAbstractsModel
from NMFMaxAbstractsModel import NMFMaxAbstractsModel
from NMFUnionAbstractsModel import NMFUnionAbstractsModel


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
        #d = DataLoader()
        #d.papers(["2013","2014","2015"]).conferences().conferenceseries().keywords()
        #self.data_tags = d.data.loc[:, ["conferenceseries", "keyword_label"]].copy()
        #self.data_tags.columns = ["conferenceseries", "tag_name"]
        file = os.path.join(".", "data", "data_tags.pkl")
        #with open(file,"wb") as f:
        #    pickle.dump(self.data_tags, f)
        with open(file,"rb") as f:
            self.data_tags = pickle.load(f)
        #del d
        self.model_tags = TagModel()
        self.model_tags.train(self.data_tags)
        self.models.append("Tags")
        #d = DataLoader()
        #d.training_data_for_abstracts("small")
        #self.data_abstracts = d.data.copy()
        file = os.path.join(".", "data", "data_abstracts.pkl")
        #with open(file,"wb") as f:
        #    pickle.dump(self.data_abstracts, f)
        with open(file,"rb") as f:
                self.data_abstracts = pickle.load(f)
        #del d
        self.model_tfidf_max = TfIdfMaxAbstractsModel()
        self.model_tfidf_max.train(data=self.data_abstracts, data_name="small")
        self.models.append("TfIdf_Max")
        self.model_tfidf_union = TfIdfUnionAbstractsModel()
        self.model_tfidf_union.train(data=self.data_abstracts, data_name="small")
        self.models.append("TfIdf_Union")
        #classifier=MultinomialNB()
        #self.model_tfidf_classifier = TFIDFClassifierAbstractsModel(classifier=classifier)
        #self.model_tfidf_classifier.train(data=self.data_abstracts, data_name="small")
        #self.models.append("TfIdf_Classifier")
        self.model_nmf = NMFAbstractsModel()
        self.model_nmf.train(data=self.data_abstracts, data_name="small")
        self.models.append("NMF_Abstracts")
        self.model_nmf_max = NMFMaxAbstractsModel()
        self.model_nmf_max.train(data=self.data_abstracts, data_name="small")
        self.models.append("NMF_Abstracts_Max")
        self.model_nmf_union = NMFUnionAbstractsModel()
        self.model_nmf_union.train(data=self.data_abstracts, data_name="small")
        self.models.append("NMF_Abstracts_Union")
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
        if modelName=="Tags":  
            rec = self.model_tags.query_batch(data)
            return self.getSeriesNames(rec)
        if modelName=="TfIdf_Max":
            rec = self.model_tfidf_max.query_single(data)
            return self.getSeriesNames(rec)
        if modelName=="TfIdf_Union":
            rec = self.model_tfidf_union.query_single(data)
            return self.getSeriesNames(rec)
        #if modelName=="TfIdf_Classifier":
            #rec = self.model_tfidf_classifier.query_single(data)
            #return self.getSeriesNames(rec)
        if modelName=="NMF_Abstracts":
            rec = self.model_nmf.query_single(data)
            return self.getSeriesNames(rec)
        if modelName=="NMF_Abstracts_Max":
            rec = self.model_nmf_max.query_single(data)
            return self.getSeriesNames(rec)
        if modelName=="NMF_Abstracts_Union":
            rec = self.model_nmf_union.query_single(data)
            return self.getSeriesNames(rec)
        print("Model not found, please select a different model")
        return False

    def autocomplete(self, modelName, data):
        if modelName == "Authors":
            return self.model_authors.get_author_names(term=data)
        if modelName == "Tags":
            return self.model_tags.get_tag_names(term=data)
    
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
            additional['Description'] = additional['Description'][:400]
            return additional
        else: 
            return None