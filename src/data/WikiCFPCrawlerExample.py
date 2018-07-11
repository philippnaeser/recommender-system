# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:20:59 2018

@author: Andreea
"""

import WikiCFPCrawler
import WikiCFPDataParser
import os

crawler = WikiCFPCrawler.WikiCFPCrawler()

##Select the range of conferences (event ids) to be crawled 
start_eventid = 0 #last crawled ID: 78661
end_eventid = 79000 

##Crawl conferences for the chosen event ids
all_conferences = crawler.crawl_conferences(start_eventid, end_eventid)

##Save crawled conferences to CSV
filename_all_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', "WikiCFP.conferences.csv"
    )
all_conferences.to_csv(filename_all_conferences)

##Parse the crawled data
#parser = WikiCFPDataParser.WikiCFPDataParser(all_conferences)

##Create pickle files for conferences in a given year
#conferences_2016 = parser.getConferencesPerYear('2016')
#conferences_2017 = parser.getConferencesPerYear('2017')
#conferences_2018 = parser.getConferencesPerYear('2018')

##Create pickle files for conferences in a given period
#conferences_2016_2017 = parser.getConferencePerPeriod('2016', '2017')
