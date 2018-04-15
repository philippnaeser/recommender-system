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

import pprint
import pickle
import os.path
import re
import time

### attributes
nt_has_conference = "<http://scigraph.springernature.com/ontologies/core/hasConference>"
nt_has_book = "<http://scigraph.springernature.com/ontologies/core/hasBook>"
nt_has_contribution = "<http://scigraph.springernature.com/ontologies/core/hasContribution>"
nt_publishedname = "<http://scigraph.springernature.com/ontologies/core/publishedName>"
nt_name = "<http://scigraph.springernature.com/ontologies/core/name>"

class FileParser:
    regex = '([<"].+?[>"])+?'
    
    path_raw = "..\\..\\data\\raw\\"
    path_persistent = "..\\..\\data\\interim\\parser\\"
    
    processes = {
        "books":{
                "filename":path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                "processLine":"processLineBooks",
                "persistentFile":path_persistent + "books.pkl",
                "persistentVariable":[]
        },
        "books_conferences":{
                "filename":path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt",
                "processLine":"processLineBooksConferences",
                "persistentFile":path_persistent + "books_conferences.pkl",
                "persistentVariable":{}
        },
        "conferences":{
                "filename":path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                "processLine":"processLineConferences",
                "persistentFile":path_persistent + "conferences.pkl",
                "persistentVariable":[]
        },
        "conferences#name":{
                "filename":path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt",
                "processLine":"processLineConferencesAttributeName",
                "persistentFile":path_persistent + "conferences#name.pkl",
                "persistentVariable":{}
        },
        "chapters_2016":{
                "filename":path_raw + "springernature-scigraph-book-chapters-2016.cc-by.2017-11-07.nt",
                "processLine":"processLineChapters",
                "persistentFile":path_persistent + "chapters_2016.pkl",
                "persistentVariable":[]
        },
        "chapters_books_2016":{
                "filename":path_raw + "springernature-scigraph-book-chapters-2016.cc-by.2017-11-07.nt",
                "processLine":"processLineChaptersBooks",
                "persistentFile":path_persistent + "chapters_books_2016.pkl",
                "persistentVariable":{}
        },
        "contributions_2016":{
                "filename":path_raw + "springernature-scigraph-book-chapters-2016.cc-by.2017-11-07.nt",
                "processLine":"processLineContributions",
                "persistentFile":path_persistent + "contributions_2016.pkl",
                "persistentVariable":[],
                "parameters":"chapters_2016"
        },
        "contributions_2016#publishedName":{
                "filename":path_raw + "springernature-scigraph-book-chapters-2016.cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsAttributePublishedName",
                "persistentFile":path_persistent + "contributions_2016#publishedName.pkl",
                "persistentVariable":{},
                "parameters":"contributions_2016"
        },
        "contributions_chapters_2016":{
                "filename":path_raw + "springernature-scigraph-book-chapters-2016.cc-by.2017-11-07.nt",
                "processLine":"processLineContributionsChapters",
                "persistentFile":path_persistent + "contributions_chapters_2016.pkl",
                "persistentVariable":{},
                "parameters":"chapters_2016"
        }
    }
    
    def __init__(self):
        self.start_time = []
        self.persistent = {}
        
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
        self.checkpoint = int(self.filesize/25)
        
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
            
    """
        Called to process file containing chapters.
    """
    def processLineChapters(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book):
            if (line[2] in self.getData("books")):
                if line[0] not in v:
                    v.append(line[0])
                    
    def processLineChaptersBooks(self,line,v,parameters):
        line = line.split()
        
        if (line[1] == nt_has_book):
            if (line[2] in self.getData("books")):
                v[line[0]] = line[2]
                
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




parser = FileParser()
testme = parser.getData("conferences#name")