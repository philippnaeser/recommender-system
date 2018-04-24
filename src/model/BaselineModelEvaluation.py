# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:23:29 2018

@author: Steff
"""
import sys
sys.path.insert(0, ".\..\data")

from BaselineModel import BaselineModel
from DataLoader import DataLoader
#import pandas as pd
#import numpy as np

"""
    Prepare the data for the evaluation.
"""

d = DataLoader()
d.papers(["2013","2014","2015"]).contributions().conferences()
data_train = d.data

model = BaselineModel()
model.train(data_train)

d = DataLoader()
d.papers(["2016"]).contributions().conferences()
data_test = d.data

model_test = BaselineModel()
model_test.train(data_test)

authors = list(model_test.get_author_names(count=0))

print("Getting recommendations.")
recommendation = model.query_batch(authors)
print("Getting truth values.")
truth = model_test.query_batch(authors)

from RecallEvaluation import RecallEvaluation

"""
    Get some evaluation scores.
"""

# add your evaluations here:

evaluation = RecallEvaluation()
evaluation.evaluate(recommendation,truth)

#evaluation = ...Evaluation()
#evaluation.evaluate(recommendation,truth)