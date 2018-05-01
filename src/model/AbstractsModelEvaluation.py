# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:39:50 2018

@author: Steff
"""

import sys
sys.path.insert(0, ".\..\data")

from AbstractsModel import AbstractsModel
from DataLoader import DataLoader
import pandas as pd

d = DataLoader()
d.papers(["2013","2014","2015"]).abstracts().conferences()
data_train = d.data

data_train = data_train[["chapter_abstract","conference","conference_name"]].copy()

data_train.drop(
        list(data_train[pd.isnull(data_train.chapter_abstract)].index),
        inplace=True
)

model = AbstractsModel()
model.train(data_train[0:20000])
#model.print_top_k(10)

test = model.query_single("Hello there, what is going on have some data for me dude")