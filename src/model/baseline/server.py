# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:38:11 2018

@author: Steff
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import pandas as pd
import json
import urllib

path = "..\\..\\..\\data\\processed\\"
data = pd.read_csv(path + "baseline.model.csv")


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        self.send_response(200)
        if self.path.startswith("/authors?"):
            self.end_headers()
            query = urllib.parse.urlparse(self.path).query
            query = urllib.parse.parse_qs(query)
            authors = bytearray(self.getAuthorNames(query["term"][0]),"utf-8")
            self.wfile.write(authors)
            
        if self.path.startswith("/author?"):
            self.end_headers()
            query = urllib.parse.urlparse(self.path).query
            query = urllib.parse.parse_qs(query)
            authors = bytearray(self.getDataByAuthor(query["author"][0]),"utf-8")
            self.wfile.write(authors)
            
        else:
            if (self.path == "/"):
                self.path = "/client.html"
                
            #self.wfile.write(b"no")
            f = open(".\\" + self.path)
            #self.send_header('Content-type','text-html')
            self.end_headers()
            html = bytearray(f.read(),"utf-8")
            self.wfile.write(html)
            f.close()

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
        
    def getAuthorNames(self,term):
        #return json.dumps(list(data["author_name"].unique()))
        authors = pd.Series(data["author_name"].unique())
        authors = authors[authors.str.lower().str.startswith(term.lower())][:10]
        return json.dumps(list(authors))
    
    def getDataByAuthor(self,author):
        return data[data["author_name"]==author].sort_values(by="count",ascending=False).to_json()
    
    def setData(self,data):
        self.data

print("Start serving.")

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()