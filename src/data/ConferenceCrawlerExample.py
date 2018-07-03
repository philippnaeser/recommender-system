# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:33:20 2018

@author: Andreea
"""

import ConferenceCrawler
import os

##URL to "All Categories" page of the WikiCFP website
categories_url = "http://www.wikicfp.com/cfp/allcat"

##Maximum number of pages to be crawled for each category
num_pages = 20

crawler = ConferenceCrawler.ConferenceCrawler(categories_url, num_pages)

all_conferences = crawler.crawl_conferences()[0]
unique_conferences = crawler.crawl_conferences()[1]

##Sort conferences by deadline
unique_conferences.sort_values('Deadline', inplace = True)

#filename_all_conferences = os.path.join(
#            os.path.dirname(os.path.realpath(__file__)),
#            "..","..","data","interim", 'WikiCFP', "all_conferences.csv"
#    )
#all_conferences.to_csv(filename_all_conferences)
#
#filename_unique_conferences = os.path.join(
#            os.path.dirname(os.path.realpath(__file__)),
#            "..","..","data","interim", 'WikiCFP', "unique_conferences.csv"
#    )
#unique_conferences.to_csv(filename_unique_conferences)