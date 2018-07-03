# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:55:15 2018

@author: Andreea
"""
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from urllib import response, parse
import time
import datetime
import pickle

class ConferenceCrawler():
    """
    Crawls conferences from the WikiCFP page by parsing the tables with
    conferences information for each category, then removing duplicates.
    
    """
    
    persistent_file_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', "conferences.pkl"
    )
    
    persistent_file_categories = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', "categories.pkl"
    )
    
    ##########################################
    def __init__(self, categories_url, num_pages):
        self.categories_url = categories_url
        self.num_pages = num_pages
        
    ##########################################
    def _parse_categories(self):
    
        category = []
        category_url = []
        
        print('Extracting the data from website.')
        
        num_pages = 1
        site = 0
        moredata = True
        
        while moredata and site < num_pages:
            site+=1
            
            #Fire the request
            try:
                print('Requesting categories search...')
                data = requests.get(self.categories_url)
                time.sleep(5)
                print("Done.")
            except Exception as e:
                moredata = False
                print(str(e))
                
            if data.status_code != 200:
                print("Can't connect to WikiCFP! (status code: " + response.status_code + ")")
                
            #Crawl the data
            soup = BeautifulSoup(data.text, 'lxml')
            idx = 0
    
            for content in soup.body.find_all('div', attrs={'class' : 'contsec'}):
                for table in content.find_all('tr', attrs={'bgcolor' : ['#f6f6f6', '#e6e6e6']}):
                    for infos in table.find_all('td'):
                        if idx == 0:
                            ##Get name and WikiCFP URL
                            category.append(infos.a.text.strip())
                            category_url.append("http://www.wikicfp.com" + infos.a['href'])
            
                        idx +=1
                        if idx==2:
                            idx=0
        
        #Convert arrays to dataframe
        categories = pd.DataFrame({'Category': category, 'Category URL': category_url})
        print('Finished parsing {} WikiCFP categories.'.format(len(categories)), "\n")
    
        return categories
    
    ##########################################
    def _parse_table(self, url):
        name = []
        detailed_name = []
        date = []
        location = []
        deadline = []
        confurl = []
        
        print('Extracting the data from website.')
        site = 0
        moredata = True
        
        parsed_url = parse.urlparse(url)
    
        while moredata and site < self.num_pages:
            site+=1
    
            #Fire the request
            try:
                print("Requesting: {}".format("".join([parsed_url.netloc, parsed_url.path, parsed_url.query, '&page=', str(site)])))
                data = requests.get(url, params = {'page': site})
                time.sleep(5) ##WikiCFP policy, issue at most one query every five seconds
                print('Done.')
            except Exception as e:
                moredata = False
                print(str(e))
            if data.status_code!= 200:
                print("Can't connect to WikiCFP! (status code: " + response.status_code + ")")
    
            #Crawl the data
            soup = BeautifulSoup(data.text, 'lxml')
    
            idx = 0
    
            for content in soup.body.find_all('div', attrs={'class' : 'contsec'}):
                for table in content.find_all('tr', attrs={'bgcolor' : ['#f6f6f6', '#e6e6e6']}):
                    for infos in table.find_all('td', attrs={'align' : 'left'}):
    
                        #Go through the table
                        if idx == 0:
                            ##Get name and WikiCFP URL
                            #print('Processing event www.wikicfp.com' + infos.a['href'])
                            name.append(infos.a.text.strip())
                            confurl.append("http://www.wikicfp.com" + infos.a['href'])
                        elif idx==1:
                            #get detailed name
                            detailed_name.append(infos.text.strip())
                        elif idx==2:
                            #Get date
                            date.append(infos.text.strip())
                        elif idx==3:
                            #Get location
                            location.append(infos.text.strip())
                        elif idx==4:
                            #Get 'Call for Papers deadline'
                            deadline.append(infos.text.strip())
                        else:
                            print("Data is unclear.")
    
                        idx +=1
                        if idx==5:
                            idx=0
        
        #Convert parsed table to dataframe
        conferences = pd.DataFrame({'Conference': name, 'Detailed Name': detailed_name, 'Date': date, 'Location': location, 'Deadline': deadline,
                                   'WikiCFP URL': confurl})
    
        print("Finished parsing tables.")
        return conferences
    
    ##########################################
    def crawl_conferences(self):
        
        if not self._load_data_categories():
            print("Categories not persistent yet. Crawling now.")
            self.categories = self._parse_categories()
            self._save_data_categories()
        
        if not self._load_data_conferences():
            print("Conferences not persistent yet. Crawling now.")
            self.all_conferences = pd.DataFrame()
            for index, row in self.categories.iterrows():
                print("Crawling conferences for category \'{}\'".format(self.categories['Category'][index]))
                
                category_url = self.categories['Category URL'][index]
                parsed_tables = self._parse_table(category_url)
                
                print('Finished crawling conferences for {}/{} categories.'.format(index+1, len(self.categories)), "\n")
                
                self.all_conferences = pd.concat([self.all_conferences, parsed_tables])
             
            ##Get unique conferences (filter by WikiCFP URL)
            self.unique_conferences = self.all_conferences.drop_duplicates(subset=['WikiCFP URL'], keep='first')
            self._save_data_conferences()

        time = datetime.datetime.now().isoformat(sep = ' ', timespec = 'seconds')
        
        print("Finished parsing {} unique WikiCFP conferences from a total of "\
              "{} conferences in {} categories.".format(len(self.unique_conferences),
              len(self.all_conferences), len(self.categories)), "\n")
        print("Finished crawling at {}.".format(time))
          
        return (self.all_conferences, self.unique_conferences)
    ##########################################
    def _save_data_categories(self):
        with open(ConferenceCrawler.persistent_file_categories,"wb") as f:
            pickle.dump(self.categories, f)
      
    ##########################################
    def _save_data_conferences(self):
        with open(ConferenceCrawler.persistent_file_conferences,"wb") as f:
            pickle.dump([self.all_conferences, self.unique_conferences], f)
    
    ##########################################  
    def _load_data_categories(self):
        if os.path.isfile(ConferenceCrawler.persistent_file_categories):
            print("Loading data: Categories")
            with open(ConferenceCrawler.persistent_file_categories,"rb") as f:
                self.categories = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    
    ##########################################
    def _load_data_conferences(self):
        if os.path.isfile(ConferenceCrawler.persistent_file_conferences):
            print("Loading data: Conferences")
            with open(ConferenceCrawler.persistent_file_conferences,"rb") as f:
                self.all_conferences, self.unique_conferences = pickle.load(f)
                print("Loaded.")
                return True
        
        return False
    