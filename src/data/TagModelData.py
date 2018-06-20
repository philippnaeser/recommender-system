# -*- coding: utf-8 -*-
"""
@author: Philipp
"""

from DataLoader import DataLoader
from FileParser import FileParser
import pandas as pd
import numpy as np

d = DataLoader()
parser = FileParser()

#We don't need all of this, but it is a convinient way to load it.
d.papers(["2013", "2014", "2015"]).conferences()
chapters_conferences = d.data.loc[:, ["chapter", "conference", "conference_name"]]

marketcodes = parser.getData("bookeditions#marketcodes")
marketcode_names = parser.getData("marketcodes#name")
all_chapters = []
all_bookeditions = []
all_marketcodes = []
chapters_13 = parser.getData("chapters_bookeditions_2013")
for key in chapters_13:
    all_chapters.append(key)
    all_bookeditions.append(chapters_13[key])
    all_marketcodes.append(marketcodes[chapters_13[key]])
chapters_14 = parser.getData("chapters_bookeditions_2014")
for key in chapters_14:
    all_chapters.append(key)
    all_bookeditions.append(chapters_14[key])
    all_marketcodes.append(marketcodes[chapters_14[key]])
chapters_15 = parser.getData("chapters_bookeditions_2015")
for key in chapters_15:
    all_chapters.append(key)
    all_bookeditions.append(chapters_15[key])
    all_marketcodes.append(marketcodes[chapters_15[key]])

df_temp = pd.DataFrame()
df_temp["chapter"] = list(all_chapters)
df_temp["bookedition"] = list(all_bookeditions)
df_temp["marketcodes"] = list(all_marketcodes)


df = pd.merge(chapters_conferences, df_temp, how="left", on=["chapter", "chapter"])

#Now that we merged this, all we need for the model are the Conference Names and the Market Codes
df = df.loc[:, ["conference_name", "marketcodes"]]

#the dict helps me to keep track of already set values
tracking = dict()
tags = []
names = []
#Now, get the dict we want, go through all rows, get the market codes and save the respective conference names
for row in df.itertuples():
    for code in row.marketcodes:
        ##Appearently, some codes do not exist anymore in the latest scigraph .nt files
        if code in marketcode_names.keys():
            if code in tracking:
                #Now, we don't want the same conference to show up multiple times for one tag
                if row.conference_name in tracking[code]:
                    continue
                else:
                    tracking[code].append(row.conference_name)
                    tags.append(marketcode_names[code])
                    names.append(row.conference_name)
            else:
                tracking[code] = [row.conference_name]
                tags.append(marketcode_names[code])
                names.append(row.conference_name)
        
df_tags = pd.DataFrame()
df_tags["tag_name"] = list(tags)
df_tags["conference_name"] = list(names)
df_tags["count"] = pd.Series(np.ones(len(df_tags)))

print(df_tags)
df_tags.to_csv("..\\..\\data\\processed\\tags.csv")

#Since this would bloat up the evaluation part, we will get the test set here as well
d = DataLoader()
d.papers(["2016"]).conferences()
chapters_conferences = d.data.loc[:, ["chapter", "conference", "conference_name"]]
parser = FileParser()
marketcodes = parser.getData("bookeditions#marketcodes")
test_chapters = []
test_bookeditions = []
test_marketcodes = []
chapters_16 = parser.getData("chapters_bookeditions_2016")
for key in chapters_16:
    test_chapters.append(key)
    test_bookeditions.append(chapters_16[key])
    test_marketcodes.append(marketcodes[chapters_16[key]])
df_temp = pd.DataFrame()
df_temp["chapter"] = list(test_chapters)
df_temp["bookedition"] = list(test_bookeditions)
df_temp["marketcodes"] = list(test_marketcodes)

df = pd.merge(chapters_conferences, df_temp, how="left", on=["chapter", "chapter"])
df = df.loc[:, ["conference_name", "marketcodes"]]

#the dict helps me to keep track of already set values
tracking = dict()
tags = []
names = []
#Now, get the dict we want, go through all rows, get the market codes and save the respective conference names
for row in df.itertuples():
    for code in row.marketcodes:
        ##Appearently, some codes do not exist anymore in the latest scigraph .nt files
        if code in marketcode_names.keys():
            if code in tracking:
                #Now, we don't want the same conference to show up multiple times for one tag
                if row.conference_name in tracking[code]:
                    continue
                else:
                    tracking[code].append(row.conference_name)
                    tags.append(marketcode_names[code])
                    names.append(row.conference_name)
            else:
                tracking[code] = [row.conference_name]
                tags.append(marketcode_names[code])
                names.append(row.conference_name)
        
df_test = pd.DataFrame()
df_test["tag_name"] = list(tags)
df_test["conference_name"] = list(names)
df_test["count"] = pd.Series(np.ones(len(df_tags)))

print(df_test)
df_test.to_csv("..\\..\\data\\processed\\tags_test.csv")