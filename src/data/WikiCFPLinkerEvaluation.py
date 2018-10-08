# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:59:50 2018

@author: Andreea
"""

###### Script parameters #######
REMOVE_STOPWORDS = True

##One of {"levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler"}.
SIMILARITY_METRIC = "damerau_levenshtein"
MATCH_THRESHOLD = 0.885

DATA_NAME = "large"
#################################


#Load the Gold Standard
from WikiCFPLinkerGoldStandard import WikiCFPLinkerGoldStandard

gs = WikiCFPLinkerGoldStandard()
gold_standard = gs.getGoldStandard()


#Load the computed correspondences
from WikiCFPLinker import WikiCFPLinker

linker = WikiCFPLinker(
        REMOVE_STOPWORDS,
        SIMILARITY_METRIC,
        MATCH_THRESHOLD, 
        DATA_NAME
        )
correspondences = linker.match_conferences()
linker.get_statistics()

#Evaluate matching
correct_predicted = 0 

for sg_series in gold_standard["scigraph_conferenceseries"]:
    if sg_series in list(correspondences["conferenceseries"]):
        predicted = correspondences[
                        correspondences["conferenceseries"] == sg_series
                        ]["WikiCFP_conferenceseries"].tolist()[0]
        truth = gold_standard[
                    gold_standard["scigraph_conferenceseries"] == sg_series
                    ]["wikicfp_conferenceseries"].tolist()[0]
        
        if predicted == truth:
            correct_predicted += 1
        
recall = correct_predicted/len(gold_standard)
precision = correct_predicted/len(correspondences)

if recall!=0 and precision!=0:
    f1_measure = 2*precision*recall/(precision+recall)
else:
    f1_measure = 0
    
print("Precision: {}, Recall: {}, F1-Measure: {}.".format(precision, recall, f1_measure))
    