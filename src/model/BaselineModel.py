# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:08:06 2018

@author: Steff
"""

from AbstractClasses import AbstractModel 
import pandas as pd

class BaselineModel(AbstractModel):
    
    #def __init__(self,data):
        #AbstractModel.__init__(self,data)
    
    def query_single(self,author):
        return self.data[self.data["author_name"]==author].sort_values(by="count",ascending=False)
        
    def train(self,data):
        """
        Set the data to be searched for by author name.
        Needs to contain 'author_name', 'conference', 'conference_name' and 'count'.
        
        Args:
            data (pandas.DataFrame): the data used by the model.
        """
        AbstractModel.train(self,data)
        
        #for check in ["author_name","conference","conference_name","count"]:
        for check in ["author_name","conference_name","count"]:
            if not check in data.columns:
                raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
        
        self.data = data
        
    def get_author_names(self,term):
        authors = pd.Series(self.data["author_name"].unique())
        authors = authors[authors.str.lower().str.startswith(term.lower())][:10]
        return authors