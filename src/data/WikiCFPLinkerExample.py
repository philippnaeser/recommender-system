# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:16:19 2018

@author: Andreea
"""
###### Script parameters #######
REMOVE_STOPWORDS = True

##One of {"levenshtein", "damerau_levenshtein", "jaro", "jaro-winkler", "monge_elkan"}.
SIMILARITY_METRIC = "jaro"
MATCH_THRESHOLD = 0.9

DATA_NAME = "small"
#################################

from WikiCFPLinker import WikiCFPLinker

linker = WikiCFPLinker(
        REMOVE_STOPWORDS,
        SIMILARITY_METRIC,
        MATCH_THRESHOLD, 
        DATA_NAME
        )
correspondences = linker.match_conferences()
linker.get_statistics()