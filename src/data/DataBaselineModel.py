# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:39:04 2018

@author: Steff
"""

import FileParser
import pandas as pd

books = parser.getData("books_conferences")
conferences_name =  parser.getData("conferences#name")
chapters = parser.getData("chapters_books_2016")
contributions = parser.getData("contributions_chapters_2016")
contributions_name = parser.getData("contributions_2016#publishedName")
                                    
chapters = pd.DataFrame(
        list(chapters.items()),
        columns=["chapter","book"]
)

conferences_by_book = pd.DataFrame(
        list(books.items()),
        columns=["book","conference"]
)

conferences_name = pd.DataFrame(
        list(conferences_name.items()),
        columns=["conference","conference_name"]
)

contributions = pd.DataFrame(
        list(contributions.items()),
        columns=["author","chapter"]
)

contributions_name = pd.DataFrame(
        list(contributions_name.items()),
        columns=["author","author_name"]
)

result = pd.merge(chapters, conferences_by_book, how="left", on=["book", "book"])
result = pd.merge(result, conferences_name, how="left", on=["conference", "conference"])
result = pd.merge(contributions, result, how="left", on=["chapter", "chapter"])
result = pd.merge(result, contributions_name, how="left", on=["author", "author"])

result["author_name"].replace(
        to_replace='"',
        value="",
        inplace=True,
        regex=True
)

result["conference_name"].replace(
        to_replace='"',
        value="",
        inplace=True,
        regex=True
)

print(result.iloc[1])

path = "..\\..\\data\\processed\\"
result.to_csv(path + "baseline.model.csv")