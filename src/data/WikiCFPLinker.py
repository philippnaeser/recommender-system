# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:20:55 2018

@author: Steff
@author: Andreea
"""

import os
import pickle
import pandas as pd
import operator
import Levenshtein as lv
import jellyfish as jf 
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
        
        self.wikicfp_conf = self._load_wikicfp_series()
        self.wikicfp_series = self.wikicfp_conf["Conference Series"].drop_duplicates()
        self.scigraph_series = self._load_scigraph_series(data_name)
        
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.stopwords = stopwords.words('english')
        
        self.match_threshold = match_threshold
        self.similarity_measure = self._get_similarity_measure(similarity_metric)
        
        self.matches = []
        self.scigraph_notmatched = list(self.scigraph_series["conferenceseries_name"])
        self.wikicfp_notmatched = list(self.wikicfp_series.values)
        self.wikicfp_names_notmatched = list(self.wikicfp_conf["Name"])
        
        self.persistent_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'linked_conferences', "matched_conference_series.pkl"
            )
        
        self.matched_conf_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'linked_conferences', "matched_conference_series.csv"
            )
        
        self.scigraph_notmatched_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'linked_conferences', "scigraph_notmatched_series.csv"
            )
        
        self.wikicfp_notmatched_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'linked_conferences', "wikicfp_notmatched_conf.csv"
            )
    
    ##########################################
    def match_conferences(self): 
        """
        Links the conference series from SciGraph to those crawled from WikiCFP.
        """
        
        if not self._load_correspondences():
            #Link conference series that exactly match between SciGraph and WikiCFP
            self._link_equally()
            
            #Link conference series that match between SciGraph and WikiCFP
            if len(self.scigraph_notmatched)!=0:
                self._link_series_to_series()
            
            #Link SciGraph conference series to WikiCFP most recent conference 
            #name belonging to a series
            if len(self.scigraph_notmatched)!=0:
                self._link_series_to_name()
             
            #Save matches as DataFrame and remove duplicates 
            #(i.e. keep matches with highest similarity score)
            self.correspondences = pd.DataFrame(
                self.matches, 
                columns = ["conferenceseries", "conferenceseries_name", 
                           "WikiCFP_conferenceseries", "similarity_score"]
                )
            self.correspondences = self.correspondences\
                                   .sort_values("similarity_score", ascending=False)\
                                .drop_duplicates(subset=["conferenceseries"])\
                                .sort_index()
    
            #Save correspondences
            self._save_correspondences()
    
            #Save remaining SciGraph conference series to be matched
            scigraph_notmatched_series = pd.DataFrame(
                    self.scigraph_notmatched,
                    columns = ["conferenceseries_name"]
                    )
            scigraph_notmatched_series.to_csv(self.scigraph_notmatched_file)
            
            #Save remaining SciGraph conference series to be matched
            wikicfp_notmatched_series = pd.DataFrame(
                    self.wikicfp_notmatched,
                    columns = ["wikicfp_conferenceseries"]
                    )
            wikicfp_notmatched_series.to_csv(self.wikicfp_notmatched_file)
                                
        return self.correspondences
    
    ##########################################
    def _link_equally(self): 
        """
        Links the conference series from SciGraph to those from WikiCFP if 
        their names are identical.
        """
        print("Computing equal matching.")
        checked = list()
        
        for series_name in self.scigraph_series["conferenceseries_name"]:
            if series_name in self.scigraph_notmatched:                
                for index in self.wikicfp_series.index.tolist():
                    wikicfp_series = self.wikicfp_series[index]
                    
                    if wikicfp_series in self.wikicfp_notmatched:
                        if self.remove_stopwords:
                            processed_wikicfp_series = self._remove_stopwords(wikicfp_series.lower())
                            processed_scigraph_series = self._remove_stopwords(series_name.lower())
                        else:
                            processed_wikicfp_series = wikicfp_series.lower()
                            processed_scigraph_series = series_name.lower()
                         
                        if processed_scigraph_series == processed_wikicfp_series:
                            similarity = 1.0
                            sg_series = self.scigraph_series[
                                    self.scigraph_series["conferenceseries_name"] == series_name
                                    ]["conferenceseries"].tolist()[0]
                            self.matches.append([sg_series, series_name, wikicfp_series, similarity])   
                            self.wikicfp_notmatched.remove(wikicfp_series)
                            checked.append(series_name)
                if series_name in checked:
                    self.scigraph_notmatched.remove(series_name)  

        print("Equal matching computed.")   
    
    ##########################################
    def _link_series_to_series(self):  
        """
        Links the conference series from SciGraph to those from WikiCFP if the 
        similarity of the conference series names, as determined by the chosen 
        similarity metric, is above the chosen threshold.
        """
        print("Computing series-to-series similarities.")
        checked = list()
        
        for series_name in self.scigraph_series["conferenceseries_name"]:
            if series_name in self.scigraph_notmatched:
                for index in self.wikicfp_series.index.tolist():
                    wikicfp_series = self.wikicfp_series[index]
                    
                    if wikicfp_series in self.wikicfp_notmatched:
                        if self.remove_stopwords:
                            processed_wikicfp_series = self._remove_stopwords(wikicfp_series.lower())
                            processed_scigraph_series = self._remove_stopwords(series_name.lower())
                        else:
                            processed_wikicfp_series = wikicfp_series.lower()
                            processed_scigraph_series = series_name.lower()
                         
                        similarity = self.similarity_measure(processed_scigraph_series, processed_wikicfp_series)
                        if similarity >= self.match_threshold:
                            sg_series = self.scigraph_series[
                                    self.scigraph_series["conferenceseries_name"] == series_name
                                    ]["conferenceseries"].tolist()[0]                            
                            self.matches.append([sg_series, series_name, wikicfp_series, similarity])   
                            self.wikicfp_notmatched.remove(wikicfp_series)
                            checked.append(series_name)
                            
                if series_name in checked:
                    self.scigraph_notmatched.remove(series_name)  

        print("Series-to-series similarities computed.")   
        
    ##########################################
    def _link_series_to_name(self):  
        """
        Links the conference series from SciGraph to the most recent conference 
        from WikiCFP that belongs to a conference series, if the similarity of 
        the SciGraph conference series name and of the WikiCFP conference name, 
        as determined by the chosen similarity metric, is above the chosen 
        threshold.
        """
        print("Computing series-to-name similarities.")
        checked = list()
        
        for series_name in self.scigraph_series["conferenceseries_name"]:
            if series_name in self.scigraph_notmatched:
                for index in self.wikicfp_conf["Name"].index.tolist():
                    wikicfp_name = self.wikicfp_conf['Name'][index]
                    
                    if wikicfp_name in self.wikicfp_names_notmatched:
                        if self.remove_stopwords:
                            processed_wikicfp_name = self._remove_stopwords(wikicfp_name.lower())
                            processed_scigraph_series = self._remove_stopwords(series_name.lower())
                        else:
                            processed_wikicfp_name = wikicfp_name.lower()
                            processed_scigraph_series = series_name.lower()
        
                        similarity = self.similarity_measure(processed_scigraph_series, processed_wikicfp_name)
                        if similarity >= self.match_threshold:
                            sg_series = self.scigraph_series[
                                    self.scigraph_series["conferenceseries_name"] == series_name
                                    ]["conferenceseries"].tolist()[0]
                            wikicfp_name = self._get_most_recent(wikicfp_name)
                            self.matches.append([sg_series, series_name, wikicfp_name, similarity])
                            checked.append(series_name)
                               
                if series_name in checked:        
                    self.scigraph_notmatched.remove(series_name)         
                         
        print("Series-to-name similarities computed.")
    
    ##########################################
    def _get_most_recent(self, conf_name): 
        """
        Returns the most similar WikiCFP conference from several WiiCFP 
        conferences with similar names.
        
        Args:
            conf_name (string): The name of the WikiCFP conference.
        
        Returns:
            string: The name of the most recent WikiCFP conference similar to
                    the given conference name.
        """
        repeating_conf= list()
        
        for index in self.wikicfp_conf["Name"].index.tolist():
            similarity = self.similarity_measure(self.wikicfp_conf["Name"][index], conf_name)
            if similarity >= self.match_threshold:
                repeating_conf.append((
                        self.wikicfp_conf["Name"][index], 
                        self.wikicfp_conf["Start Date"][index]
                        ))
        most_recent = max(repeating_conf, key=operator.itemgetter(1))[0]   
        
        for wikicfp_name in [elem[0] for elem in repeating_conf]:
            if wikicfp_name in self.wikicfp_names_notmatched:
                self.wikicfp_names_notmatched.remove(wikicfp_name)
            
        return most_recent  
        
    ##########################################
    def get_statistics(self):
        """
        Prints statistics of the matched conference series.
        """
        
        print("There are {} conference series in WikiCFP.".format(
                len(self.wikicfp_series)))
        print("There are {} conference series in the SciGraph considered data.".format(
                len(self.scigraph_series)))
        
        percentage_matched = len(self.correspondences)/len(self.scigraph_series)
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
        return similarity
    
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
        return similarity
    
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
        return similarity

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
        return similarity
    
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
        scigraph_series = scigraph_series.drop_duplicates()
                                
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
                ][["Conference", "Name", "Conference Series", "Start Date"]]
        
        return wikicfp_conf
    
    ##########################################
    def _save_correspondences(self):
        """
        Saves the computed correspondences between the SciGraph and the 
        WikiCFP conference series.
        """
        self.correspondences.to_csv(self.matched_conf_file)
        
        print("Saving to disk.")
        with open(self.persistent_file,"wb") as f:
            pickle.dump(self.correspondences, f)
    
    ##########################################
    def _load_correspondences(self):
        """
        Loads the computed correspondences between the SciGraph and the 
        WikiCFP conference series.
        """
        if os.path.isfile(self.persistent_file):
            print("Loading correspondences.")
            with open(self.persistent_file,"rb") as f:
                self.correspondences = pickle.load(f)
                print("Loaded.")
                return True
        
        return False