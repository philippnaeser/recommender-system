# -*- coding: utf-8 -*-
"""
Created on Thu May 10 01:45:24 2018

@author: Steff
"""

import pickle

path_raw = "..\\..\\data\\raw\\"
filename = path_raw + "springernature-scigraph-conferences.cc-zero.2017-11-07-UPDATED.nt"
nt_has_conference = "<http://scigraph.springernature.com/ontologies/core/hasConference>"
nt_name = "<http://scigraph.springernature.com/ontologies/core/name>"

count = 0
filesize = 0
checkpoint = 0

try:
    with open("..\\..\\data\\interim\\exploration\\multiple_conferences.pkl","rb") as f:
        multiple_conferences = pickle.load(f)
except FileNotFoundError:
    def count_lines(filename):
        print("Start counting lines.")
        global filesize
        global count
        global checkpoint
        
        c = 0
        with open(filename) as f:
            for line in f:
                c += 1
        
        filesize = c
        count = 0
        checkpoint = int(filesize/100)
    
        print("Finished counting lines: {}".format(filesize))
        
    def increaseCount():
        global count
        global checkpoint
        
        count += 1
        if (count % checkpoint == 0):
            print("Checkpoint reached: {}%".format(int(count*100/filesize)))
    
    multiple_conferences = {}
    
    count_lines(filename)
    with open(filename) as f:
        for line in f:
            increaseCount()
            line = line.split()
            if (line[1] == nt_name):
                if line[0].startswith("<http://scigraph.springernature.com/things/conferences/"):
                    try:
                        multiple_conferences[line[0]] += 1
                    except KeyError:
                        multiple_conferences[line[0]] = 1
                        
    with open("..\\..\\data\\interim\\exploration\\multiple_conferences.pkl","wb") as f:
        pickle.dump(multiple_conferences, f)
    
hist = {}

for val in multiple_conferences:
    try:
        hist[multiple_conferences[val]] += 1
    except KeyError:
        hist[multiple_conferences[val]] = 1
        
print(hist)