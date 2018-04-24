# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:08:06 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import pandas as pd
import numpy as np

class BaselineModel(AbstractModel):
    
    #def __init__(self,data):
        #AbstractModel.__init__(self,data)
    
    def query_single(self,author):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            author (str): The author name.
        
        Returns:
            str[]: id of the conference
            str[]: name of the conference
            double[]: confidence scores
        """
        if not isinstance(author,str):
            raise TypeError("argument 'author' needs to be a string.")

        data = self.data[self.data["author_name"]==author].sort_values(by="count",ascending=False)
        conference = list(data["conference_name"])
        confidence = list(data["count"])

        return [conference,confidence]
   
    def query_batch(self,batch):
        """
        Queries the model and returns a list of recommendations for each request.
        
        Args:
            batch[str]: The list of author names.
        
        Returns:
            A list of size 'len(batch)' which contains the recommendations for each item of the batch.
            If author not found, the value is None.
            
            str[]: name of the conference
            double[]: confidence scores
        """
        if not isinstance(batch,list):
            raise TypeError("argument 'batch' needs to be a list of author names.")
        
        data = self.data[self.data["author_name"].isin(batch)]
        
        count = len(data)
        checkpoint = max(int(count/20),1)
        i = 0
        recommendations = {}
        for index, row in data.iterrows():
            if (i%checkpoint)==0:
                print("Batch querying: {}%".format(int(i/count*100)))
            
            author = row["author_name"]
            conference = row["conference_name"]
            value = row["count"]
            
            try:
                recommendations[author][conference] = value
            except KeyError:
                recommendations[author] = {conference:value}
                
            i += 1
        
        
        conference = list()
        confidence = list()
        for q in batch:
            try:
                conference.append(
                        sorted(recommendations[q], key=recommendations[q].__getitem__, reverse=True)
                )
                confidence.append(
                        sorted(recommendations[q].values(),reverse=True)
                )
            except KeyError:
                conference.append(None)
                confidence.append(None)
        
            
        return [conference,confidence]
    
    def train(self,data):
        """
        Set the data to be searched for by author name.
        Needs to contain 'author_name' and 'conference_name'.
        
        Args:
            data (pandas.DataFrame): the data used by the model.
        """
        AbstractModel.train(self,data)
        
        #for check in ["author_name","conference","conference_name","count"]:
        for check in ["author_name","conference_name"]:
            if not check in data.columns:
                raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
        
        data["count"] = pd.Series(np.ones(len(data)))
        self.data = data[["author_name","conference_name","count"]].groupby(by=["author_name","conference_name"]).sum().reset_index()
        #self.data.drop_duplicates(inplace=True)
        
    def get_author_names(self,term="",count=10):
        authors = pd.Series(self.data["author_name"].unique())
        
        if count>0:
            authors = authors[authors.str.lower().str.startswith(term.lower())][:count]
        else:
            authors = authors[authors.str.lower().str.startswith(term.lower())]
        
        return authors