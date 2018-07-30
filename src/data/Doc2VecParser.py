# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:55:51 2018

@author: Andreea
"""

import os
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

class Doc2VecParser():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "doc_embeddings"
        )
    
    filepaths = {
            "d2v_100d_w5_HS":os.path.join(path_persistent,"d2v_100d_w5_HS"),
            "d2v_100d_w5_HS_2":os.path.join(path_persistent,"d2v_100d_w5_HS_2"),
            "d2v_100d_w5_NS_2":os.path.join(path_persistent,"d2v_100d_w5_NS_2")
            }
        
    #################################################    
    def load_model(self, model):
        try:
            self.model = Doc2Vec.load(self.filepaths[model])
        except KeyError:
            print("This model does not exist.")
    #################################################    
    def transform_vector(self, document):
        """
        Transform a document into a vector containing embeddings.
        
        Args:
            document (str): The document to be transformed.
            
        Returns:
            A numpy array that contains embeddings.
        """
        vector = self.model.infer_vector(
                word_tokenize(document.lower())
                )
        
        return vector

    #################################################
    def transform_vectors(self, documents):
        """
        Transform a list of strings into a list of vectors containing embeddings.
        
        Args:
            documents list(str): The list of documents to be transformed.
            
        Returns:
            A list of numpy arrays that contain embeddings.
        """
        vectors = list()
        for document in documents:
            transformed_document = self.model.infer_vector(
                            word_tokenize(document.lower())
                            ) 
            vectors.append(transformed_document)
        return vectors
            
        
        
        
        
        