# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:20:55 2018

@author: Steff
"""

import pandas as pd
from WikiCFPCrawler import WikiCFPCrawler

crawler = WikiCFPCrawler()
crawler._load_conferences()

not_nones = crawler.all_conferences[
        ~pd.isnull(crawler.all_conferences["Conference Series"])
]

from DataLoader import DataLoader

d = DataLoader()
d.papers([2016]).conferences().conferenceseries()
samples = d.data[["conferenceseries","conferenceseries_name"]]\
            .drop_duplicates()\
            .sample(frac=1)
samples = samples[0:100]
#samples.to_csv("conferenceseries_to_match.csv")