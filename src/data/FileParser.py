# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:38:11 2018

@author: Steff
"""

"""

BookChapter
    <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://scigraph.springernature.com/ontologies/core/BookChapter>
    <http://scigraph.springernature.com/ontologies/core/hasContribution>
    <http://scigraph.springernature.com/ontologies/core/title>
    <http://scigraph.springernature.com/ontologies/core/hasBook>
    
Book
    <http://scigraph.springernature.com/ontologies/core/hasConference>

"""

import pickle
import os.path
import re
import time

### shared attributes
nt_name = "<http://scigraph.springernature.com/ontologies/core/name>"

### book attributes
nt_has_conference = "<http://scigraph.springernature.com/ontologies/core/hasConference>"

### chapter attributes
nt_has_book = "<http://scigraph.springernature.com/ontologies/core/hasBook>"
nt_has_book_edition = "<http://scigraph.springernature.com/ontologies/core/hasBookEdition>"
nt_has_contribution = "<http://scigraph.springernature.com/ontologies/core/hasContribution>"
nt_abstract = "<http://scigraph.springernature.com/ontologies/core/abstract>"
nt_title = "<http://scigraph.springernature.com/ontologies/core/title>"
nt_language = "<http://scigraph.springernature.com/ontologies/core/language>"

## bookedition attributes
nt_has_productmarketcode = "<http://scigraph.springernature.com/ontologies/core/hasProductMarketCode>"

## contributions attributes
nt_publishedname = "<http://scigraph.springernature.com/ontologies/core/publishedName>"
nt_iscorresponding = "<http://scigraph.springernature.com/ontologies/core/isCorresponding>"
nt_order = "<http://scigraph.springernature.com/ontologies/core/order>"

## conference attributes
nt_acronym = "<http://scigraph.springernature.com/ontologies/core/acronym>"
nt_city = "<http://scigraph.springernature.com/ontologies/core/city>"
nt_country = "<http://scigraph.springernature.com/ontologies/core/country>"
nt_dateend = "<http://scigraph.springernature.com/ontologies/core/dateEnd>"
nt_datestart = "<http://scigraph.springernature.com/ontologies/core/dateStart>"
nt_year = "<http://scigraph.springernature.com/ontologies/core/year>"
nt_has_conference_series = "<http://scigraph.springernature.com/ontologies/core/hasConferenceSeries>"

class FileParser:
    regex = '([<"].*?[>"])+?'
    
    path_raw = "..\\..\\data\\raw\\"
    path_persistent = "..\\..\\data\\interim\\parser\\"
    
    years = list(range(2011,2018)) + [
                "2009-2010",
                "2006-2008",
                "2001-2005",
                "1996-2000",
                "1991-1995",
                "1981-1990",
                "1971-1980",
                "1951-1970",
                "1901-1950",
                "1801-1900"
            ]
    
    def __init__(self):
        self.start_time = []
        self.persistent = {}
        
        self.processes = {
            "books":{
                    "filename":self.path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                    "processLine":"processLineBooks",
                    "persistentFile":self.path_persistent + "books.pkl",
                    "persistentVariable":[]
            },
            "books_conferences":{
                    "filename":self.path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                    "processLine":"processLineBooksConferences",
                    "persistentFile":self.path_persistent + "books_conferences.pkl",
                    "persistentVariable":{}
            },
            ### bookeditions
            "bookeditions":{
                    "filename":self.path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                    "processLine":"processLineBookEditions",
                    "persistentFile":self.path_persistent + "bookeditions.pkl",
                    "persistentVariable":[]
            },
            ### bookeditions#marketcodes
            "bookeditions#marketcodes":{
                    "filename":self.path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                    "processLine":"processLineBookEditionsAttributeMarketCodes",
                    "persistentFile":self.path_persistent + "bookeditions#marketcodes.pkl",
                    "persistentVariable":{}
            },
            "conferences":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferences",
                    "persistentFile":self.path_persistent + "conferences.pkl",
                    "persistentVariable":[]
            },
            "conferences#name":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeName",
                    "persistentFile":self.path_persistent + "conferences#name.pkl",
                    "persistentVariable":{}
            },
            "conferences#acronym":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeAcronym",
                    "persistentFile":self.path_persistent + "conferences#acronym.pkl",
                    "persistentVariable":{}
            },
            "conferences#city":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeCity",
                    "persistentFile":self.path_persistent + "conferences#city.pkl",
                    "persistentVariable":{}
            },
            "conferences#country":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeCountry",
                    "persistentFile":self.path_persistent + "conferences#country.pkl",
                    "persistentVariable":{}
            },
            "conferences#dateend":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeDateEnd",
                    "persistentFile":self.path_persistent + "conferences#dateend.pkl",
                    "persistentVariable":{}
            },
            "conferences#datestart":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeDateStart",
                    "persistentFile":self.path_persistent + "conferences#datestart.pkl",
                    "persistentVariable":{}
            },
            "conferences#year":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesAttributeYear",
                    "persistentFile":self.path_persistent + "conferences#year.pkl",
                    "persistentVariable":{}
            },
            "conferenceseries":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferenceseries",
                    "persistentFile":self.path_persistent + "conferenceseries.pkl",
                    "persistentVariable":[]
            },
            "conferenceseries#name":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferenceseriesAttributeName",
                    "persistentFile":self.path_persistent + "conferenceseries#name.pkl",
                    "persistentVariable":{}
            },
            "conferences_conferenceseries":{
                    "filename":self.path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                    "processLine":"processLineConferencesConferenceseries",
                    "persistentFile":self.path_persistent + "conferences_conferenceseries.pkl",
                    "persistentVariable":{}
            }
        }
        
        # initialize processes of yearly data
        for year in FileParser.years:
            year = str(year)
            ### chapters
            self.processes["chapters_" + year] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineChapters",
                "persistentFile":self.path_persistent + "chapters_" + year + ".pkl",
                "persistentVariable":[]
            }
            ### chapters#abstract
            self.processes["chapters_" + year + "#abstract"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by-nc.2017-11-07.nt",
                "processLine":"processLineChaptersAttributeAbstract",
                "persistentFile":self.path_persistent + "chapters_" + year + "#abstract.pkl",
                "persistentVariable":{},
                "parameters":"chapters_" + year
            }
            ### chapters#title
            self.processes["chapters_" + year + "#title"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineChaptersAttributeTitle",
                "persistentFile":self.path_persistent + "chapters_" + year + "#title.pkl",
                "persistentVariable":{},
                "parameters":"chapters_" + year
            }
            ### chapters#language
            self.processes["chapters_" + year + "#language"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineChaptersAttributeLanguage",
                "persistentFile":self.path_persistent + "chapters_" + year + "#language.pkl",
                "persistentVariable":{},
                "parameters":"chapters_" + year
            }
            ### chapters_books
            self.processes["chapters_books_" + year] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineChaptersBooks",
                "persistentFile":self.path_persistent + "chapters_books_" + year + ".pkl",
                "persistentVariable":{}
            }
            ### chapters_bookeditions
            self.processes["chapters_bookeditions_" + year] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineChaptersBookEditions",
                "persistentFile":self.path_persistent + "chapters_bookeditions_" + year + ".pkl",
                "persistentVariable":{},
                "parameters":"chapters_" + year
            }
            ### contributions
            self.processes["contributions_" + year] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineContributions",
                "persistentFile":self.path_persistent + "contributions_" + year + ".pkl",
                "persistentVariable":[],
                "parameters":"chapters_" + year
            }
            ### contributions#publishedName
            self.processes["contributions_" + year + "#publishedName"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsAttributePublishedName",
                "persistentFile":self.path_persistent + "contributions_" + year + "#publishedName.pkl",
                "persistentVariable":{},
                "parameters":"contributions_" + year
            }
            ### contributions#isCorresponding
            self.processes["contributions_" + year + "#isCorresponding"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsAttributeIsCorresponding",
                "persistentFile":self.path_persistent + "contributions_" + year + "#isCorresponding.pkl",
                "persistentVariable":{},
                "parameters":"contributions_" + year
            }
            ### contributions#order
            self.processes["contributions_" + year + "#order"] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsAttributeOrder",
                "persistentFile":self.path_persistent + "contributions_" + year + "#order.pkl",
                "persistentVariable":{},
                "parameters":"contributions_" + year
            }
            ### contributions_chapters
            self.processes["contributions_chapters_" + year] = {
                "filename":self.path_raw + "springernature-scigraph-book-chapters-" + year + ".cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsChapters",
                "persistentFile":self.path_persistent + "contributions_chapters_" + year + ".pkl",
                "persistentVariable":{},
                "parameters":"chapters_" + year
        }
        
    ### start runtime check
    def tic(self):
        self.start_time.append(time.time())
    
    ### print runtime information
    def toc(self):
        print("--- %s seconds ---" % (time.time() - self.start_time.pop()))
    
    def getData(self,process):
        ### is the data already present?
        if (process in self.persistent):
            return self.persistent[process]
        
        print("Process '{}' not in memory yet.".format(process))
        
        ### load from persistence if already processed
        if os.path.isfile(self.processes[process]["persistentFile"]):
            with open(self.processes[process]["persistentFile"],"rb") as f:
                self.persistent[process] = pickle.load(f)
                return self.persistent[process]
            
        print("Process '{}' not persistent yet. Processing.".format(process))
        
        ### get the data from scratch
        self.persistent[process] = self.processes[process]["persistentVariable"]
        self.parseFile(
                self.processes[process]["filename"],
                self.processes[process]["processLine"],
                self.persistent[process],
                self.processes[process]["parameters"] if "parameters" in self.processes[process] else None
        )
        with open(self.processes[process]["persistentFile"],"wb") as f:
            pickle.dump(self.persistent[process], f)
        
        return self.persistent[process]
    
    def parseFile(self,filename,processLine,variable,parameters):
        self.countLines(filename)
        self.processFile(filename,processLine,variable,parameters)
        
    """
        Count the lines upfront to provide progress.
    """
    def countLines(self,filename):
        print("Start counting lines.")
        self.tic()
        
        count = 0
        with open(filename) as f:
            for line in f:
                count += 1
        
        self.filesize = count
        self.count = 0
        self.checkpoint = int(self.filesize/100)
        
        self.toc()
        print("Finished counting lines: {}".format(self.filesize))
    
    """
        Prints information about the progress.
    """
    def increaseCount(self):
        self.count += 1
        if (self.count % self.checkpoint == 0):
            print("Checkpoint reached: {}%".format(int(self.count*100/self.filesize)))
    
    """
        Process a given file calling function @process for each line.
        @process is given @variable to store results.
    """
    def processFile(self,filename,processLine,variable,parameters):
        print("Start processing file.")
        self.tic()
        
        processLineFunction = self.__getattribute__(processLine)
        
        ### 14sec / 23sec split / 77sec regex
        with open(filename) as f:
            for line in f:
                self.increaseCount()
                processLineFunction(line,variable,parameters)
        
        self.toc()
        print("Finished processing file.")
    
    ################## Process implementations ##################
    
    """
        Called to process file containing books.
    """
    def processLineBooks(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_conference):
            if line[0].startswith("<http://scigraph.springernature.com/things/books/"):
                if line[0] not in v:
                    v.append(line[0])
                
    def processLineBooksConferences(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_conference):
            if line[0].startswith("<http://scigraph.springernature.com/things/books/"):
                v[line[0]] = line[2]
                
    """
        Called to process file containing conferences.
    """
    def processLineConferences(self,line,v,parameters):
        line = line.split()
        
        if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
            if line[0] not in v:
                v.append(line[0])
                
    def processLineConferencesAttributeName(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_name):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
            
    def processLineConferencesAttributeAcronym(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_acronym):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferencesAttributeCity(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_city):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferencesAttributeCountry(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_country):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferencesAttributeDateEnd(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_dateend):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferencesAttributeDateStart(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_datestart):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferencesAttributeYear(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_year):
            if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                v[line[0]] = line[2]
                
    def processLineConferenceseries(self,line,v,parameters):
        line = line.split()
        
        if line[0].startswith("<http://scigraph.springernature.com/things/conference-series/"):
            if line[0] not in v:
                v.append(line[0])
                
    def processLineConferencesConferenceseries(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_conference_series):
            v[line[0]] = line[2]
            
    def processLineConferenceseriesAttributeName(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_name):
            if line[0].startswith("<http://scigraph.springernature.com/things/conference-series/"):
                v[line[0]] = line[2]
    
                
    """
        Called to process file containing chapters.
    """
    def processLineChapters(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book):
            if (line[2] in self.getData("books")):
                if line[0] not in v:
                    v.append(line[0])
                    
    def processLineChaptersAttributeTitle(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_title):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]
                
    def processLineChaptersAttributeLanguage(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_language):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]

    def processLineChaptersBooks(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book):
            if (line[2] in self.getData("books")):
                v[line[0]] = line[2]
                
    def processLineChaptersBookEditions(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book_edition):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]
                                
    def processLineBookEditions(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book_edition):
            if (line[0] in self.getData("books")):
                if line[2] not in v:
                    v.append(line[2])
                    
    def processLineBookEditionsAttributeMarketCodes(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_productmarketcode):
            if (line[0] in self.getData("bookeditions")):
                if line[0] not in v:
                    v[line[0]] = []
                if line[2] not in v[line[0]]:
                    v[line[0]].append(line[2])
   
    def processLineContributions(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_contribution):
            if (line[0] in self.getData(parameters)):
                if line[2] not in v:
                    v.append(line[2])
                    
    def processLineContributionsChapters(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_contribution):
            if (line[0] in self.getData(parameters)):
                v[line[2]] = line[0]
                
    def processLineContributionsAttributePublishedName(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_publishedname):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]

    def processLineContributionsAttributeIsCorresponding(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_iscorresponding):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]
                
    def processLineContributionsAttributeOrder(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_order):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]
                
    """
        Called to process file containing abstracts.
    """
    def processLineChaptersAttributeAbstract(self,line,v,parameters):
        line = re.findall(self.regex, line)
        
        if (line[1] == nt_abstract):
            if (line[0] in self.getData(parameters)):
                v[line[0]] = line[2]
    

    
    

        
        
                 
                 

"""
### contribution attributes
nt_family_name = "<http://scigraph.springernature.com/ontologies/core/publishedFamilyName>"
nt_given_name = "<http://scigraph.springernature.com/ontologies/core/publishedGivenName>"

nt_corresponding = "<http://scigraph.springernature.com/ontologies/core/isCorresponding>"
nt_order = "<http://scigraph.springernature.com/ontologies/core/order>"

attributes_contribution = [
        nt_family_name,
        nt_given_name,
        nt_name,
        nt_corresponding,
        nt_order
]
"""



#parser = FileParser()

#parser.getData("contributions_2014#publishedName")
#parser.getData("contributions_2015#publishedName")
#parser.getData("contributions_2017#publishedName")
#parser.getData("contributions_chapters_2014")
#parser.getData("contributions_chapters_2015")
#parser.getData("contributions_chapters_2017")
#parser.getData("chapters_books_2014")
#parser.getData("chapters_books_2015")
#parser.getData("chapters_books_2017")

#parser.getData("chapters_2014#title")
#parser.getData("chapters_2015#title")
#parser.getData("chapters_2016#title")
#parser.getData("chapters_2017#title")
               
#parser.getData("chapters_2014#language")
#parser.getData("chapters_2015#language")
#parser.getData("chapters_2016#language")
#parser.getData("chapters_2017#language")
               
#parser.getData("conferences#acronym")
#parser.getData("conferences#city")
#parser.getData("conferences#country")
#parser.getData("conferences#dateend")
#parser.getData("conferences#datestart")
#parser.getData("conferences#year")

#parser.getData("conferences_conferenceseries")               
#parser.getData("conferenceseries")
#parser.getData("conferenceseries#name")

#parser.getData("contributions_2014#isCorresponding")
#parser.getData("contributions_2015#isCorresponding")
#parser.getData("contributions_2016#isCorresponding")
#parser.getData("contributions_2017#isCorresponding")
#parser.getData("contributions_2014#order")
#parser.getData("contributions_2015#order")
#parser.getData("contributions_2016#order")
#parser.getData("contributions_2017#order")

#parser.getData("chapters_bookeditions_2014")
#parser.getData("chapters_bookeditions_2015")
#parser.getData("chapters_bookeditions_2016")
#parser.getData("chapters_bookeditions_2017")
#parser.getData("bookeditions")
#parser.getData("bookeditions#marketcodes")

#parser.getData("chapters_2013")
#parser.getData("chapters_books_2013")
#parser.getData("chapters_bookeditions_2013")
#parser.getData("chapters_2013#title")
#parser.getData("chapters_2013#language")
#parser.getData("chapters_2013#abstract")
#parser.getData("contributions_2013")
#parser.getData("contributions_2013#publishedName")
#parser.getData("contributions_2013#isCorresponding")
#parser.getData("contributions_2013#order")
#parser.getData("contributions_chapters_2013")

def parseYear(year):
    parser.getData("chapters_" + year)
    parser.getData("chapters_books_" + year)
    parser.getData("chapters_bookeditions_" + year)
    parser.getData("chapters_" + year + "#title")
    parser.getData("chapters_" + year + "#language")
    parser.getData("chapters_" + year + "#abstract")
    parser.getData("contributions_" + year)
    parser.getData("contributions_" + year + "#publishedName")
    parser.getData("contributions_" + year + "#isCorresponding")
    parser.getData("contributions_" + year + "#order")
    parser.getData("contributions_chapters_" + year)
    
#parseYear("2012")
#parseYear("2011")

#years = [
#        "2009-2010",
#        "2006-2008",
#        "2001-2005",
#        "1996-2000",
#        "1991-1995",
#        "1981-1990",
#        "1971-1980",
#        "1951-1970",
#        "1901-1950",
#        "1801-1900"
#    ]

#for y in years:
#    parseYear(y)