# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 08:48:56 2018

@author: Andreea
"""

###### Script parameters #######
MATCHED_CONFERENCES = "matched_conference_series.pkl"

THRESHOLD_DATE = "2018-07-11"
#################################

#Load the SciGraph-WikiCFP matches
import os
import pickle

persistent_file = os.path.join(
            "..","..","data","interim", 'linked_conferences', 
            MATCHED_CONFERENCES
            )
if os.path.isfile(persistent_file):
    with open(persistent_file,"rb") as f:
        print("Loading correspondences.")
        matched_conferences = pickle.load(f)
        print("... loaded.")
sg_conferences = matched_conferences["conferenceseries"]
        
#Run WikiCFPSearcher
from WikiCFPSearcher import WikiCFPSearcher
searcher = WikiCFPSearcher(THRESHOLD_DATE)
wikicfp_data = searcher.retrieve_info(sg_conferences)
print("Number of conferences with submission deadline after {}: {}.".format(
        THRESHOLD_DATE, len(wikicfp_data.keys())))