# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

from AbstractClasses import AbstractModel 
import pandas as pd
import numpy as np

class TagModel(AbstractModel):
    
    ##########################################
    def query_single(self,tag, recs=10):
        """
        Queries the model and returns a list of recommendations.
        
        Args:
            tag (str): The tag name.
        
        Returns:
            str[]: name of the conference
            double[]: confidence scores
        """
        if not isinstance(tag,str) and not isinstance(tag, list):
            raise TypeError("argument 'tag' needs to be a string or a list of strings.")
        data = []
        conference = list()
        confidence = list()
        if isinstance(tag, str):
            data = self.data[self.data["tag_name"]==tag].sort_values(by="count",ascending=False)
            conference = list(data["conferenceseries"])
            confidence = list(data["count"])
        else:
            for t in tag:
                data = self.data[self.data["tag_name"]==t].sort_values(by="count",ascending=False)
                for conf in list(data["conferenceseries"]):
                    if not conf in conference:
                        conference.append(conference)
                        confidence.append(1)
                    else:
                        # increase the count
                        index = conference.index(conf)
                        confidence[index] = confidence[index]+1
                temp = pd.DataFrame()
                temp["conferenceseries"] = conference
                temp["count"] = confidence
                temp.sort_values(by="count", ascending=False)
                conference = list(temp["conferenceseries"])
                confidence = list(temp["count"])

        return [conference[0:recs],confidence[0:recs]]
    ##########################################
    def query_batch(self,batch, recs=10):
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
                
        count = len(batch)
        checkpoint = max(int(count/20),1)
        i = 0
        conference = list()
        confidence = list()
        for q in batch:
            if (i%checkpoint)==0:
                print("Batch querying: {}%".format(int(i/count*100)))   
            i += 1
            query = self.query_single(q, recs)
            conference.append(query[0])
            confidence.append(query[1])
        
            
        return [conference,confidence]
    ###########################################
    def train(self,data):
        """
        Set the data to be searched for by tag name.
        Needs to contain 'tag_name' and 'conferenceseries'.
        
        Args:
            data (pandas.DataFrame): the data used by the model.
        """
        AbstractModel.train(self,data)
        
        for check in ["tag_name","conferenceseries"]:
            if not check in data.columns:
                raise IndexError("Column '{}' not contained in given DataFrame.".format(check))
        
        data["count"] = pd.Series(np.ones(len(data)))
        self.data = data[["tag_name","conferenceseries","count"]].groupby(by=["tag_name","conferenceseries"]).sum().reset_index()
        #self.data.drop_duplicates(inplace=True)
        ##########################################
    def get_tag_names(self,term="",count=10):
        """
        Returns the first 'count' number of author names starting with the string 'term'.
        If count=0, then all author names starting with 'term' will be returned.
        
        Args:
            term (str): String the author name starts with.
            count (int): The number of authors to return.
            
        Returns:
            A list of author names (strings).
        """
        tags = pd.Series(self.data["tag_name"].unique())
        
        if count>0:
            tags = tags[tags.str.lower().str.startswith(term.lower())][:count]
        else:
            tags = tags[tags.str.lower().str.startswith(term.lower())]
        
        return tags