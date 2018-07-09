# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:52:16 2018

@author: Andreea
"""
import pandas as pd
import os
import pickle

class WikiCFPDataParser():
    
    filename = "WikiCFP.period.conferences.pkl"
    
    persistent_file_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', filename)
        
    ##########################################
    def __init__(self, all_conferences):
        self.all_conferences = all_conferences
        
    ##########################################
    def displayDescription(self, conferences):
        """
        Prints description of crawled conferences.
        
        Args:
            conferences (dataFrame): crawled conferences
            
        """      
        for event in range(len(conferences)):
            description = conferences.iloc[event]['Description']
    
            if description:
                for line in description.split('\n'):
                    print(line)
            else:
                print("This conference has no description.")
                
            print("################################")
            
    ##########################################   
    def getConferencesPerYear(self, year):
        """
        Returns the conferences which take place in the given year.
        
        Args:
            year (string): year in which confereces take place
            
        Returns:
            dataFrame: crawled conferences taking place in the given year
        """      
        
        conferences_in_year = pd.DataFrame()
        if self.all_conferences['Start Date'].astype(str).str.contains(year).any():
            conferences_in_year = self.all_conferences[self.all_conferences['Start Date'].astype(str).str.contains(year)]
            
            persistent_file_conferences = self._changePath(str(year))
            self._save_conferences(conferences_in_year ,persistent_file_conferences)
        else:
            print("There are no conferences taking place in {}.\n".format(year))
        
        return conferences_in_year
        
    ##########################################   
    def getConferencePerPeriod(self, start_period, end_period):
        """
        Returns the conferences which take place in a given period (i.e. multiple years).
        
        Args:
            start_period (string): the starting year of the period in which 
                confereces take place
            end_period (string): the ending year of the period in which 
                confereces take place
            
        Returns:
            dataFrame: crawled conferences taking place in the given period
        """      
        
        conferences_in_period = pd.DataFrame()
        
        period = list(range(int(start_period), int(end_period)+1))
        index = 0
        
        while index < len(period): 
            conferences_in_year = pd.DataFrame()
            year = period[index]

            if self.all_conferences['Start Date'].astype(str).str.contains(str(year)).any():
                conferences_in_year = self.all_conferences[self.all_conferences['Start Date'].astype(str).str.contains(str(year))]
                
            if not conferences_in_year.empty:
                conferences_in_period = pd.concat([conferences_in_period, conferences_in_year])
            
            index += 1
            
        if not conferences_in_period.empty:
            period = "".join([start_period, "-", end_period])
            
            persistent_file_conferences = self._changePath(str(period))
            self._save_conferences(conferences_in_period, persistent_file_conferences)
        else:
            print("There are no conferences taking place in the period {}-{}.\n".format(start_period, end_period))
        
        return conferences_in_period

    ##########################################   
    def _save_conferences(self, conferences, persistent_file_conferences):
        with open(persistent_file_conferences, "wb") as f:
            pickle.dump(conferences, f)
    
    ##########################################   
    def _changePath(self, period):
        """
        Changes the file path to the persistent file where data can be saved 
        based on the given period for which the conferences are crawled.
        
        Args:
            period (string): the period in which the crawled confereces take place
            
        Returns:
            path: the changed persisten path
        """      
        filename = self._replacePathPart(WikiCFPDataParser.persistent_file_conferences, period)[0]
        persistent_file_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","interim", 'WikiCFP', filename)
    
        return persistent_file_conferences
    
    ##########################################
    def _splitPath(self, path): 
        """
        Splits a file path in multiple parts.
        
        Args:
            path (string): the file path
            
        Returns:
            list: last parts of file path
        """      
        allparts = [] 
        while 1: 
            parts = os.path.split(path) 
            if parts[0] == path: # sentinel for absolute paths 
                allparts.insert(0, parts[0]) 
                break 
            elif parts[1] == path: # sentinel for relative paths 
                allparts.insert(0, parts[1]) 
                break 
            else: 
                path = parts[0] 
                allparts.insert(0, parts[1]) 
                return allparts
        
    ##########################################
    def _replacePathPart(self, path, period):
        """
        Replaces part of the file path with given string.
        
        Args:
            path (string): the file path
            period (string): string to replace the chosen part of the file path
            
        Returns:
            string: replaced file path part
        """      
        part = self._splitPath(path)
        if len(part)==1:
            part[0] = part[0].replace('period', period)
        
        return part