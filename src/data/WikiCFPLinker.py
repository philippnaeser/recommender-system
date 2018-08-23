# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:20:55 2018

@author: Steff
@author: Andreea
"""

#crawler = WikiCFPCrawler()
#crawler._load_conferences()

#not_nones = crawler.all_conferences[
#        ~pd.isnull(crawler.all_conferences["Conference Series"])
#]
#
#from DataLoader import DataLoader
#
#d = DataLoader()
#d.papers([2016]).conferences().conferenceseries()
#samples = d.data[["conferenceseries","conferenceseries_name"]]\
#            .drop_duplicates()\
#            .sample(frac=1)
#samples = samples[0:100]
##samples.to_csv("conferenceseries_to_match.csv")

import os
import pickle
import pandas as pd
import Levenshtein as lv
import jellyfish as jf 
import textdistance
from nltk.corpus import stopwords
from WikiCFPCrawler import WikiCFPCrawler
from DataLoader import DataLoader

class WikiCFPLinker():
    
    ##########################################
    def __init__(self, remove_stopwords = True, 
                 similarity_metric = "jaro", 
                 match_threshold = 0.9, 
                 data_name = "small"):
        
        self.crawler = WikiCFPCrawler()
        self.wikicfp_series = self._load_wikicfp_series()
        self.scigraph_series = self._load_scigraph_series(data_name)
        
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.stopwords = stopwords.words('english')
        
        self.match_value = match_threshold
        self.similarity_measure = self._get_similarity_measure(similarity_metric)
        
        self.matches = []
        self.checked = list()
        
        self.persistent_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', "matched_conference_series.pkl"
            )
        
#        self.temp_file = os.path.join(
#            os.path.dirname(os.path.realpath(__file__)),
#            "..","..","data","interim", 'WikiCFP', "matched_conference_series.csv"
#            )
    
    ##########################################
    def match_conferences(self):  
        print("Computing similarities.")
        
        for series in self.scigraph_series["conferenceseries_name"]:
            series_id = self.scigraph_series[
                            self.scigraph_series["conferenceseries_name"] == series
                            ].index.tolist()[0]
            
            for index in self.wikicfp_series.index.tolist():
                if series_id not in self.checked:
                    
                    wikicfp_series = self.wikicfp_series[index]
                    
                    if self.remove_stopwords:
                        processed_wikicfp_series = self._remove_stopwords(wikicfp_series.lower())
                        processed_scigraph_series = self._remove_stopwords(series.lower())
                    else:
                        processed_wikicfp_series = wikicfp_series.lower()
                        processed_scigraph_series = series.lower()
                     
                    if self.similarity_measure(processed_scigraph_series, processed_wikicfp_series):
                        self._update_matches(series_id, series, wikicfp_series)
            
        print("Similarities computed.")
        self.correspondences = pd.DataFrame(
                self.matches, 
                columns = ["conferenceseries", "conferenceseries_name", "WikiCFP_conferenceseries"]
                )
#        self.correspondences.to_csv(self.temp_file)
        self._save_correspondences()

        return self.correspondences
    
    ##########################################
    def get_statistics(self):
        """
        Prints statistics of the matched conference series.
        """
        
        print("There are {} conference series in WikiCFP.".format(len(self.wikicfp_series)))
        print("There are {} conference series in the SciGraph considered data.".format(len(self.scigraph_series)))
        
        percentage_matched = len(self.checked)/len(self.scigraph_series)
        print("{} out of {} conference series have been matched, i.e. {}".format(
                len(self.correspondences), len(self.scigraph_series), percentage_matched))
        
        
    ##########################################
    def _remove_stopwords(self, string):
        """
        Removes stopwords from a given string.
        
        Args:
            string (string): The string from which to remove the stopwords.
        
        Returns:
            string: The string without stopwords.
        """
        string = ' '.join([word for word in string.split() if word not in self.stopwords])
        return string

    ##########################################
    def _get_similarity_measure(self, similarity_metric):
        """
        Returns the similarity measure given the chosen metric.
        
        Args:
            similarity_metric (string): The similarity metric to be used.
        
        Returns:
            method: The chosen similarity measure. 
        """
        metric_name = "_" + str(similarity_metric) + "_match"
        similarity_measure = getattr(self, metric_name)
        return similarity_measure

    ##########################################
    def _update_matches(self, series_id, series, wikicfp_series):
        """
        Updates the correspondences and checked conference series lists.
        
        Args:
            series_id (int): The ID of the SciGraph conference series.
            wikicfp_series (string): The WikiCFP conference series name.
        """
        self.matches.append([series_id, series, wikicfp_series])   
        self.checked.append(series_id)
    
    ##########################################
    def _levenshtein_match(self, string1, string2):
        """
        Computes the Levenshtein similarity between two strings.
        
        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.
        
        Returns:
            boolean: True, if the similarity is above a given thereshold, 
            false otherwise.
        """
        similarity = lv.ratio(string1, string2)
        return similarity>=self.match_value
    
    ##########################################
    def _damerau_levenshtein_match(self, string1, string2):
        """
        Computes the Damerau Levenshtein similarity between two strings.
        
        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.
        
        Returns:
            boolean: True, if the similarity is above a given thereshold, 
            false otherwise.
        """
        distance = jf.damerau_levenshtein_distance(string1, string2)
        similarity = 1-distance/max(len(string1), len(string2))
        return similarity>=self.match_value
    
    ##########################################
    def _jaro_match(self, string1, string2):
        """
        Computes the Jaro similarity between two strings.
        
        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.
        
        Returns:
            boolean: True, if the similarity is above a given thereshold, 
            false otherwise.
        """
        similarity = lv.jaro(string1, string2)
        return similarity>=self.match_value

    ##########################################
    def _jaro_winkler_match(self, string1, string2):
        """
        Computes the Jaro-Winkler similarity between two strings.
        
        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.
        
        Returns:
            boolean: True, if the similarity is above a given thereshold, 
            false otherwise.
        """
        similarity = jf.jaro_winkler(string1, string2)
        return similarity>=self.match_value
    
    ##########################################
    def _monge_elkan_match(self, string1, string2):
        """
        Computes the Monge-Elkan similarity between two strings.
        
        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.
        
        Returns:
            boolean: True, if the similarity is above a given thereshold, 
            false otherwise.
        """
        jaro_winkler_similarity = textdistance.monge_elkan(string1, string2)
        return jaro_winkler_similarity>=self.match_value
    
    ##########################################
    def _load_scigraph_series(self, data_name):
        """
        Loads the training and test data for the given data size.
        
        Args:
            data_name (string): The size of the data to be loaded.
        
        Returns:
            dataFrame: The concatenated conference series for the training and
            test data.
        """
        d_train = DataLoader()
        d_train.training_data(data_name)
        training_conference_series = d_train.data[["conferenceseries", "conferenceseries_name"]]
                                
        d_test = DataLoader()
        d_test.test_data(data_name)
        test_conference_series = d_test.data[["conferenceseries", "conferenceseries_name"]]
        
        scigraph_series = pd.concat([training_conference_series, test_conference_series])
        scigraph_series.drop_duplicates(inplace = True)
                                
        return scigraph_series
    
    ##########################################
    def _load_wikicfp_series(self):
        """
        Loads the WikiCFP conference series.
        
        Returns:
            dataFrame: The WikiCFP conference series.
        """
        self.crawler._load_conferences()
        wikicfp_conf = self.crawler.all_conferences[
                ~pd.isnull(self.crawler.all_conferences["Conference Series"])
                ][["Conference", "Name", "Conference Series"]]
        wikicfp_series = wikicfp_conf["Conference Series"].drop_duplicates()
        
        return wikicfp_series
    
    ##########################################
    def _save_correspondences(self):
        """
        Saves the computed correspondences between the SciGraph and the 
        WikiCFP conference series.
        """
        print("Saving to disk.")
        with open(self.persistent_file,"wb") as f:
            pickle.dump(self.correspondences, f)
    
    ##########################################
    def _load_correspondences(self):
        if os.path.isfile(self.persistent_file):
            print("Loading correspondences.")
            with open(self.persistent_file,"rb") as f:
                self.correspondences = pickle.load(f)
                print("Loaded.")
                return True
        
        return False