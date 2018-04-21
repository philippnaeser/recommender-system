# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 03:16:42 2018

@author: Steff
"""

import DataLoader

d = DataLoader.DataLoader()
"""
    Functions return self, so calls can be concatenated.
    
    Possible functions:
        - abstracts: adds abstracts
        - contributions: adds authors + attributes (name, order, corresponding)
        - conferences: adds conferences + attributes (acronym, city, country, dateend, datestart, name, year)
        - conferenceseries: adds conferenceseries + attributes (name)
        - papers: adds papers + attributes (book#id, title, language)
        
    The range of years needs to be given at an appropriate call:
        papers, abstracts or contributions
        
    Calls are left merges, so the order of the calls matter.
    
    Examples
    ------------------------------------
        Abstracts in 2016:
            d.abstracts(["2016"])
        All Conferences:
            d.conferences()
        Papers in 2015 with conferences:
            d.papers(["2015"]).conferences()
        Papers in 2015, 2016 with abstracts, contributions, conferences and conferenceseries
            d.papers(["2015","2016"]).abstracts().contributions().conferences().conferenceseries()
"""

#d.abstracts(["2016"])
#d.conferences()
#d.papers(["2015"]).conferences()
d.papers(["2015","2016"]).abstracts().contributions().conferences().conferenceseries()

print(d.data)

#d.data.to_csv("test.csv")