# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 23:57:37 2018

@author: Steff
"""

import FileParser
import pandas as pd
import pickle
import os

class DataLoader:
    
    path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..","..","data","processed"
    )
    
    #path = FileParser.FileParser.path_persistent
    
    def __init__(self):
        self.parser = FileParser.FileParser()
      
    # add papers
    def papers(self, years=None):
        if hasattr(self,"years") and years is not None:
            raise AttributeError("years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self,"years"):
            raise AttributeError("years needed.")

        data = None
        
        for y in self.years:
            df_chapters = pd.DataFrame(
                    list(self.parser.getData("chapters_books_" + y).items()),
                    columns=["chapter","book"]
            )
            df_title = pd.DataFrame(
                    list(self.parser.getData("chapters_" + y + "#title").items()),
                    columns=["chapter","chapter_title"]
            )
            df_language = pd.DataFrame(
                    list(self.parser.getData("chapters_" + y + "#language").items()),
                    columns=["chapter","chapter_language"]
            )
            
            df = pd.merge(df_chapters, df_title, how="left", on=["chapter", "chapter"])
            df = pd.merge(df, df_language, how="left", on=["chapter", "chapter"])
            
            if data is None:
                data = df
            else:
                data = pd.concat([data,df])
        
        data.chapter_language = data.chapter_language.str[1:-1]
        data.chapter_title = data.chapter_title.str[1:-1]
        data = data[data.chapter_language=="En"]
        
        if hasattr(self,"data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data, how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("needs papers.")
        else:
            self.data = data
        
        return self
            
            
    # add abstracts
    def abstracts(self, years=None):
        if hasattr(self,"years") and years is not None:
            raise ValueError("years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self,"years"):
            raise AttributeError("years needed.")
            
        data = None
            
        for y in self.years:
            df = pd.DataFrame(
                    list(self.parser.getData("chapters_" + y + "#abstract").items()),
                    columns=["chapter","chapter_abstract"]
            )
            
            if data is None:
                data = df
            else:
                data = pd.concat([data,df])
        
        data.chapter_abstract = data.chapter_abstract.str[9:-1]
        
        if hasattr(self,"data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data, how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("needs papers.")
        else:
            self.data = data

        return self
    
    
    # load conferences
    def conferences(self):
        if not hasattr(self,"data"):
            df_conferences = pd.DataFrame(
                    self.parser.getData("conferences"),
                    columns=["conference"]
            )

        elif "book" in self.data.keys():
            df_conferences = pd.DataFrame(
                    list(self.parser.getData("books_conferences").items()),
                    columns=["book","conference"]
            )
        else:
            raise KeyError("Needs papers.")
        
        df_acronym = pd.DataFrame(
                list(self.parser.getData("conferences#acronym").items()),
                columns=["conference","conference_acronym"]
        )
        df_city = pd.DataFrame(
                list(self.parser.getData("conferences#city").items()),
                columns=["conference","conference_city"]
        )
        df_country = pd.DataFrame(
                list(self.parser.getData("conferences#country").items()),
                columns=["conference","conference_country"]
        )
        df_dateend = pd.DataFrame(
                list(self.parser.getData("conferences#dateend").items()),
                columns=["conference","conference_dateend"]
        )
        df_datestart = pd.DataFrame(
                list(self.parser.getData("conferences#datestart").items()),
                columns=["conference","conference_datestart"]
        )
        df_name = pd.DataFrame(
                list(self.parser.getData("conferences#name").items()),
                columns=["conference","conference_name"]
        )
        df_year = pd.DataFrame(
                list(self.parser.getData("conferences#year").items()),
                columns=["conference","conference_year"]
        )
            
        df = pd.merge(df_conferences, df_acronym, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_city, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_country, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_dateend, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_datestart, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_name, how="left", on=["conference", "conference"])
        df = pd.merge(df, df_year, how="left", on=["conference", "conference"])
        
        df.conference_name = df.conference_name.str[1:-1]
        df.conference_datestart = df.conference_datestart.str[1:-1]
        df.conference_dateend = df.conference_dateend.str[1:-1]
        df.conference_year = df.conference_year.str[1:-1]
        df.conference_city = df.conference_city.str[1:-1]
        df.conference_country = df.conference_country.str[1:-1]
        df.conference_acronym = df.conference_acronym.str[1:-1]

        if hasattr(self,"data"):
            self.data = pd.merge(self.data, df, how="left", on=["book", "book"])
        else:
            self.data = df
        
        return self
    
    
    # load contributions
    def contributions(self, years=None):
        if hasattr(self,"years") and years is not None:
            raise ValueError("years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self,"years"):
            raise AttributeError("years needed.")
            
        if not hasattr(self,"data"):
            data = None
            
            for y in self.years:
                df_contribution = pd.DataFrame(
                    list(self.parser.getData("contributions_" + y).items()),
                    columns=["contribution"]
                )
                
                if data is None:
                    data = df_contribution
                else:
                    data = pd.concat([data,df_contribution])
        
        elif "chapter" in self.data.keys(): 
            data = None
                
            for y in self.years:
                df_contribution = pd.DataFrame(
                        list(self.parser.getData("contributions_chapters_" + y).items()),
                        columns=["contribution","chapter"]
                )
                
                if data is None:
                    data = df_contribution
                else:
                    data = pd.concat([data,df_contribution])
        
        else:
            raise KeyError("Needs papers.")
        
        df = None
        for y in self.years:
            df_publishedName = pd.DataFrame(
                    list(self.parser.getData("contributions_" + y + "#publishedName").items()),
                    columns=["contribution","author_name"]
            )
            if df is None:
                df = df_publishedName
            else:
                df = pd.concat([df,df_publishedName])
        data = pd.merge(data, df, how="left", on=["contribution", "contribution"])
        
        df = None
        for y in self.years:
            df_order = pd.DataFrame(
                    list(self.parser.getData("contributions_" + y + "#order").items()),
                    columns=["contribution","author_order"]
            )
            if df is None:
                df = df_order
            else:
                df = pd.concat([df,df_order])
        data = pd.merge(data, df, how="left", on=["contribution", "contribution"])
        
        df = None
        for y in self.years:
            df_isCorresponding = pd.DataFrame(
                    list(self.parser.getData("contributions_" + y + "#isCorresponding").items()),
                    columns=["contribution","author_corresponding"]
            )
            if df is None:
                df = df_isCorresponding
            else:
                df = pd.concat([df,df_isCorresponding])
        data = pd.merge(data, df, how="left", on=["contribution", "contribution"])
            
        data.author_name = data.author_name.str[1:-1]
        data.author_order = data.author_order.str[1:-1]
        data.author_corresponding = data.author_corresponding.str[1:-1]

        if hasattr(self,"data"):
            self.data = pd.merge(self.data, data, how="right", on=["chapter", "chapter"])
        else:
            self.data = data
        
        return self
    
    
    # load conferenceseries
    def conferenceseries(self):
        if not hasattr(self,"data"):
            df_conferenceseries = pd.DataFrame(
                    self.parser.getData("conferenceseries"),
                    columns=["conferenceseries"]
            )

        elif "conference" in self.data.keys():
            df_conferenceseries = pd.DataFrame(
                    list(self.parser.getData("conferences_conferenceseries").items()),
                    columns=["conference","conferenceseries"]
            )
        else:
            raise KeyError("Needs conferences.")
            
        df_name = pd.DataFrame(
                list(self.parser.getData("conferenceseries#name").items()),
                columns=["conferenceseries","conferenceseries_name"]
        )
          
        df = pd.merge(df_conferenceseries, df_name, how="left", on=["conferenceseries", "conferenceseries"])
        
        df.conferenceseries_name = df.conferenceseries_name.str[1:-1]
        
        if hasattr(self,"data"):
            self.data = pd.merge(self.data, df, how="left", on=["conference", "conference"])
        else:
            self.data = df
            
        return self
    
    
    
    def make_persistent(self, filename):
        file = os.path.join(self.path,filename)
        with open(file,"wb") as f:
            pickle.dump(self.data, f)
    
    
    
    def get_persistent(self, filename):
        try:
            file = os.path.join(self.path,filename)
            with open(file,"rb") as f:
                self.data = pickle.load(f)
                return True
        except FileNotFoundError:
            return False