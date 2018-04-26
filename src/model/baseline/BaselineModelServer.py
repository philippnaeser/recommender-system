# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:38:11 2018

@author: Steff
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
#from io import BytesIO
import pandas as pd
import json
import urllib
import sys
import os

sys.path.insert(0, ".\..")
import BaselineModel
current_dir = os.getcwd()
sys.path.insert(0, ".\..\..\data")
from DataLoader import DataLoader

class BaselineModelHTTPRequestHandler(BaseHTTPRequestHandler):
    
    #def __init(self)
    
    def get_variable(self,name):
        """
        Returns a GET variable contained in the path of the request.
        E.g.: 'authors?term=Heiko' returns 'Heiko' if name='term'.
        
        Args:
            name (str): Name of the GET variable to return.
            
        Returns:
            str: Value of the GET variable.
            
        """
        query = urllib.parse.urlparse(self.path).query
        query = urllib.parse.parse_qs(query)
        return query[name][0]
    
    def do_GET(self):
        self.model = model
        
        if self.path.startswith("/authors?"):
            self.send_response(200)
            self.end_headers()
            
            term = self.get_variable("term")
            
            authors = self.model.get_author_names(term)
            authors = json.dumps(list(authors))
            authors = bytearray(authors,"utf-8")
            
            self.wfile.write(authors)
            
        elif self.path.startswith("/author?"):
            self.send_response(200)
            self.end_headers()
            
            query = self.get_variable("author")
            
            conferences = self.model.query_single(query)
            conferences = json.dumps(conferences)
            conferences = bytearray(conferences,"utf-8")
            
            self.wfile.write(conferences)
            
        else:
            if (self.path == "/"):
                self.path = "/client.html"
            
            #SimpleHTTPRequestHandler.do_GET(self)
            
            try:
                f = open(".\\" + self.path)
                self.send_response(200)
                self.end_headers()
                html = bytearray(f.read(),"utf-8")
                self.wfile.write(html)
                f.close()
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            

    """
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        response.write(b'This is POST request. ')
        response.write(b'Received: ')
        response.write(body)
        self.wfile.write(response.getvalue())
    """


print("Setting up server.")

handler = BaselineModelHTTPRequestHandler
httpd = HTTPServer(('localhost', 8000), handler)

#path = "..\\..\\..\\data\\processed\\"
#data = pd.read_csv(path + "baseline.model.csv")

os.chdir(".\..\..\data")
d = DataLoader()
d.papers(["2013","2014","2015"]).contributions().conferences()
os.chdir(current_dir)
data_train = d.data

model = BaselineModel.BaselineModel()
model.train(data_train)

print("Start serving.")
httpd.serve_forever()