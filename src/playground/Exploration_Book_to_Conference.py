# -*- coding: utf-8 -*-
"""
Created on Thu May 10 01:45:24 2018

@author: Steff
"""

import pickle

path_raw = "..\\..\\data\\raw\\"
filename = path_raw + "springernature-scigraph-books.cc-by.2017-11-07.nt"
nt_has_conference = "<http://scigraph.springernature.com/ontologies/core/hasConference>"

count = 0
filesize = 0
checkpoint = 0

try:
    with open("..\\..\\data\\interim\\exploration\\conference_counts.pkl","rb") as f:
        conference_counts = pickle.load(f)
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
    
    conference_counts = {}
    
    count_lines(filename)
    with open(filename) as f:
        for line in f:
            increaseCount()
            line = line.split()
            if (line[1] == nt_has_conference):
                if line[0].startswith("<http://scigraph.springernature.com/things/books/"):
                    try:
                        conference_counts[line[0]] += 1
                    except KeyError:
                        conference_counts[line[0]] = 1
                        
    with open("..\\..\\data\\interim\\exploration\\conference_counts.pkl","wb") as f:
        pickle.dump(conference_counts, f)
    
hist = {}

for val in conference_counts:
    try:
        hist[conference_counts[val]] += 1
    except KeyError:
        hist[conference_counts[val]] = 1
        
print(hist)