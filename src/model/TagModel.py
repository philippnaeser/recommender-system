# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

from AbstractClasses import AbstractModel 
import pandas as pd
import numpy as np

class TagModel(AbstractModel):
    
    ##########################################
    def query_single(self,tag):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            tag (str): The tag name.
        
        Returns:
            str[]: name of the conference
            double[]: confidence scores
        """
        if not isinstance(tag,str):
            raise TypeError("argument 'tag' needs to be a string.")

        data = self.data[self.data["tag_name"]==tag].sort_values(by="count",ascending=False)
        conference = list(data["conference_name"])
        confidence = list(data["count"])

        return [conference,confidence]
    ##########################################
    def query_batch(self,batch):
        """
        Queries the model and returns a list of recommendations for each request.
        
        Args:
            batch[str]: The list of tag names.
        
        Returns:
            A list of size 'len(batch)' which contains the recommendations for each item of the batch.
            If tag not found, the value is None.
            
            str[]: name of the conference
            double[]: confidence scores
        """
        if not isinstance(batch,list):
            raise TypeError("argument 'batch' needs to be a list of tag names.")
        
        data = self.data[self.data["tag_name"].isin(batch)]
        
        count = len(data)
        checkpoint = max(int(count/20),1)
        i = 0
        recommendations = {}
        for index, row in data.iterrows():
            if (i%checkpoint)==0:
                print("Batch querying: {}%".format(int(i/count*100)))
            
            tag = row["tag_name"]
            conference = row["conference_name"]
            value = row["count"]
            
            try:
                recommendations[tag][conference] = value
            except KeyError:
                recommendations[tag] = {conference:value}
                
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
    ###########################################
    def train(self,data):
        """
        Set the data to be searched for by tag name.
        Needs to contain 'tag_name' and 'conference_name'.
        
        Args:
            data (pandas.DataFrame): the data used by the model.
        """
        AbstractModel.train(self,data)
        
        for check in ["tag_name","conference_name"]:
            if not check in data.columns:
                raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
        
        data["count"] = pd.Series(np.ones(len(data)))
        self.data = data[["tag_name","conference_name","count"]].groupby(by=["tag_name","conference_name"]).sum().reset_index()
        #self.data.drop_duplicates(inplace=True)
        ##########################################
    def get_tag_names(self,term="",count=10):
        """
        Returns the first 'count' number of tag names starting with the string 'term'.
        If count=0, then all tag names starting with 'term' will be returned.
        
        Args:
            term (str): String the tag name starts with.
            count (int): The number of tag to return.
            
        Returns:
            A list of tag names (strings).
        """
        tags = pd.Series(self.data["tag_name"].unique())
        
        if count>0:
            tags = tags[tags.str.lower().str.startswith(term.lower())][:count]
        else:
            tags = tags[tags.str.lower().str.startswith(term.lower())]
        
        return tags