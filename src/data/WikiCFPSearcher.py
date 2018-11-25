# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:27:27 2018

@author: Andreea
"""

import os
import pickle
from collections import defaultdict
from WikiCFPCrawler import WikiCFPCrawler
from WikiCFPLinker import WikiCFPLinker

class WikiCFPSearcher():
    
    ##########################################
    def __init__(self, threshold_date = "2018-07-11"):
        
        self.threshold_date = threshold_date
        self.crawler = WikiCFPCrawler()
        self.linker = WikiCFPLinker()        
        
        self.wikicfp_conf = self._load_wikicfp_conferences()
        self.wikicfp_series = self.wikicfp_conf['Conference Series'].drop_duplicates()
        self.correspondences = self._load_correspondences()
        self.wikicfp_data = defaultdict()
        
        self.persistent_file = os.path.join(
            "..","..","data","interim", 'linked_conferences', "WikiCFP_data.pkl"
            )
        
    ##########################################
    def retrieve_info(self, sg_conferences):
        """
        Retrieves the WikiCFP data corresponding to a list of SciGraph
        conference series.
            
        Args:
            sg_conferences (list): List of SciGraph conference series.
        
        Returns:
            dict[]: dictionary with (keys=SciGraph conference series,
                    values = WikiCFp data for the chosen fields).
        """
        if not self._load_wikicfp_data():
            self._search_correspondence(sg_conferences)
            self._save_wikicfp_data()
            
        return self.wikicfp_data

    ##########################################
    def _search_correspondence(self, sg_conferences):
        """
        Searches through the SciGraph-WikiCFP conferences to find a 
        correspondence. If found, retrieves the corresponding WikiCFP data.
            
        Args:
            sg_conferences (list): List of SciGraph conference series.
        """
        for sg_conf in sg_conferences:
            if sg_conf in list(self.correspondences["conferenceseries"]):
                match = self.correspondences[
                                self.correspondences['conferenceseries']==sg_conf
                                ]['WikiCFP_conferenceseries'].tolist()[0]
                
                if match in list(self.wikicfp_series):
                    confID = self._get_latest_conference(match)
                else:
                    confID = self._get_conference(match)
                
                if confID is not None:
                    info = self._get_info(confID)
                    if self._check_period(info["Submission Deadline"]):
                        self.wikicfp_data[sg_conf] = info
    
    ##########################################
    def _get_info(self, confID):
        """
        Retrieves the WikiCFP data corresponding to a conference ID.
            
        Args:
            conference (int): A WikiCFP conference ID.
        
        Returns:
            dict[]: Dictionary containing the WikiCFP data.
        """
        
        acronym = self.wikicfp_conf.loc[confID]["Conference"]
        name = self.wikicfp_conf.loc[confID]["Name"]
        conf_series = self.wikicfp_conf.loc[confID]["Conference Series"]
        start_date = self.wikicfp_conf.loc[confID]["Start Date"]
        end_date = self.wikicfp_conf.loc[confID]["End Date"]
        location = self.wikicfp_conf.loc[confID]["Location"]
        abstract_deadline = self.wikicfp_conf.loc[confID]["Abstract Deadline"]
        submission_deadline = self.wikicfp_conf.loc[confID]["Submission Deadline"]
        notification_due = self.wikicfp_conf.loc[confID]["Notification Due"]
        final_version_deadline = self.wikicfp_conf.loc[confID]["Final Version Deadline"]
        categories = self.wikicfp_conf.loc[confID]["Categories"]
        description = self.wikicfp_conf.loc[confID]["Description"]
        external_link = self.wikicfp_conf.loc[confID]["Link"]
        wikicfp_link = "http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid="\
                        + str(confID)
        
        info = {
                "Acronym": acronym,
                "Name": name,
                "Conference Series": conf_series,
                "Start Date": start_date,
                "End Date": end_date,
                "Location": location,
                "Abstract Deadline": abstract_deadline,
                "Submission Deadline": submission_deadline,
                "Notification Due": notification_due,
                "Final Version Due": final_version_deadline,
                "Categories": categories,
                "Description": description,
                "External Link": external_link,
                "WikiCFP Link": wikicfp_link
                }
        
        
        return info
    
    ##########################################
    def _get_conference(self, conference):
        """
        Retrieves the ID of a given conference.
            
        Args:
            conference(str): A WikiCFP conference name
        
        Returns:
            int[]: The ID of the conference.
        """
        index = self.wikicfp_conf[
                self.wikicfp_conf["Name"]==conference.encode("utf-8").decode("unicode_escape")
                ].index.tolist()[0]
        
        return index
    
    ##########################################
    def _get_latest_conference(self, conference_series):
        """
        Retrieves the ID of the most recent conference from the given
        conference series.
            
        Args:
            conference_series (str): A WikiCFP conference series name.
        
        Returns:
            int[]: The ID of most recent conference.
        """
        dates = self.wikicfp_conf[
                    self.wikicfp_conf["Conference Series"] == conference_series
                    ]["Start Date"].tolist()
        dates = [date for date in dates if date is not None]
        if dates:
            most_recent_date = max(dates)
            index = self.wikicfp_conf[
                        (self.wikicfp_conf['Conference Series'] == conference_series)
                        &
                        (self.wikicfp_conf["Start Date"] == most_recent_date)
                        ].index.tolist()[0]
        else:
            index = None
        
        return index
    
    ##########################################
    def _check_period(self, submission_deadline):
        """
        Verifies whether the starting date of a conference is equal or bigger
        than the threshold date.
            
        Args:
            start_date (str): The starting date of a particular conference.
        
        Returns:
            boolean: True, if the conference is taking place later than the 
                    chosen thresold date
                    False, otherwise.
        """
        if submission_deadline is None:
            return False
        else:
            return submission_deadline >= self.threshold_date
    
    ##########################################
    def _save_wikicfp_data(self):
        with open(self.persistent_file,"wb") as f:
            pickle.dump(self.wikicfp_data, f)
            
    ##########################################
    def _load_wikicfp_data(self):
        if os.path.isfile(self.persistent_file):
            with open(self.persistent_file,"rb") as f:
                print("Loading WikiCFP data.")
                self.wikicfp_data = pickle.load(f)
                print("... loaded.")
                return True
        
        return False
       
    ##########################################
    def _load_wikicfp_conferences(self):
        """
        Loads the WikiCFP conferences.
        
        Returns:
            dataFrame: The WikiCFP conferences.
        """
        self.crawler._load_conferences()
        wikicfp_conf = self.crawler.all_conferences
        
        return wikicfp_conf
    
    ##########################################
    def _load_correspondences(self):
        """
        Loads the computed correspondences between the SciGraph and the 
        WikiCFP conference series.
        
        Returns:
            dataFrame: The SciGraph-WikiCFP correspondences.
        """
        self.linker._load_correspondences()
        correspondences = self.linker.correspondences
        
        return correspondences