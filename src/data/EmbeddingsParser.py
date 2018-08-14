# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 20:30:27 2018

@author: Steff
@author: Andreea
"""

import os
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import spacy
import numpy as np

class EmbeddingsParser:

    path_pretrained_embeddings = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "external"
    )

    path_embeddings = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "word_embeddings"
    )

    
    paths = {
            "6d50":os.path.join(path_pretrained_embeddings,"glove.6B.50d.txt"),
            "6d50-w2v":os.path.join(path_pretrained_embeddings,"glove.6B.50d-w2v.txt"),
            "6d50-folder":os.path.join(path_pretrained_embeddings,"glove.6B.50d-spacy",""),
            "6d100":os.path.join(path_pretrained_embeddings,"glove.6B.100d.txt"),
            "6d100-w2v":os.path.join(path_pretrained_embeddings,"glove.6B.100d-w2v.txt"),
            "6d100-folder":os.path.join(path_pretrained_embeddings,"glove.6B.100d-spacy",""),
            "6d200":os.path.join(path_pretrained_embeddings,"glove.6B.200d.txt"),
            "6d200-w2v":os.path.join(path_pretrained_embeddings,"glove.6B.200d-w2v.txt"),
            "6d200-folder":os.path.join(path_pretrained_embeddings,"glove.6B.200d-spacy",""),
            "6d300":os.path.join(path_pretrained_embeddings,"glove.6B.300d.txt"),
            "6d300-w2v":os.path.join(path_pretrained_embeddings,"glove.6B.300d-w2v.txt"),
            "6d300-folder":os.path.join(path_pretrained_embeddings,"glove.6B.300d-spacy",""),
            "42d300":os.path.join(path_pretrained_embeddings,"glove.42B.300d.txt"),
            "42d300-w2v":os.path.join(path_pretrained_embeddings,"glove.42B.300d-w2v.txt"),
            "42d300-folder":os.path.join(path_pretrained_embeddings,"glove.42B.300d-spacy",""),
            "840d300":os.path.join(path_pretrained_embeddings,"glove.840B.300d.txt"),
            "840d300-w2v":os.path.join(path_pretrained_embeddings,"glove.840B.300d-w2v.txt"),
            "840d300-folder":os.path.join(path_pretrained_embeddings,"glove.840B.300d-spacy",""),
            "word2vec-w2v":os.path.join(path_pretrained_embeddings,"GoogleNews-vectors-negative300.bin"),
            "word2vec-folder":os.path.join(path_pretrained_embeddings,"word2vec-spacy",""),
            "fasttext-w2v": os.path.join(path_pretrained_embeddings, "wiki-news-300d-1M.vec"),
            "fasttext-folder":os.path.join(path_pretrained_embeddings, "fasttext-spacy",""),
            "w2v_50d_w5_CBOW_HS-w2v":os.path.join(path_embeddings, "w2v_50d_w5_CBOW_HS.bin"),
            "w2v_50d_w5_CBOW_HS-folder":os.path.join(path_embeddings, "w2v_50d_w5_CBOW_HS-spacy",""),
            "w2v_50d_w5_CBOW_NS-w2v":os.path.join(path_embeddings, "w2v_50d_w5_CBOW_NS.bin"),
            "w2v_50d_w5_CBOW_NS-folder":os.path.join(path_embeddings, "w2v_50d_w5_CBOW_NS-spacy",""),
            "w2v_50d_w10_SG_HS-w2v":os.path.join(path_embeddings, "w2v_50d_w10_SG_HS.bin"),
            "w2v_50d_w10_SG_HS-folder":os.path.join(path_embeddings, "w2v_50d_w10_SG_HS-spacy",""),
            "w2v_50d_w10_SG_NS-w2v":os.path.join(path_embeddings, "w2v_50d_w10_SG_NS.bin"),
            "w2v_50d_w10_SG_NS-folder":os.path.join(path_embeddings, "w2v_50d_w10_SG_NS-spacy",""),
            "w2v_100d_w10_SG_NS-w2v":os.path.join(path_embeddings, "w2v_100d_w10_SG_NS.bin"),
            "w2v_100d_w10_SG_NS-folder":os.path.join(path_embeddings, "w2v_100d_w10_SG_NS-spacy",""),
            "w2v_100d_w10_SG_HS-w2v":os.path.join(path_embeddings, "w2v_100d_w10_SG_HS.bin"),
            "w2v_100d_w10_SG_HS-folder":os.path.join(path_embeddings, "w2v_100d_w10_SG_HS-spacy",""),
            "w2v_150d_w10_SG_NS-w2v":os.path.join(path_embeddings, "w2v_150d_w10_SG_NS.bin"),
            "w2v_150d_w10_SG_NS-folder":os.path.join(path_embeddings, "w2v_150d_w10_SG_NS-spacy",""),
            "w2v_300d_w10_SG_NS-w2v":os.path.join(path_embeddings, "w2v_300d_w10_SG_NS.bin"),
            "w2v_300d_w10_SG_NS-folder":os.path.join(path_embeddings, "w2v_300d_w10_SG_NS-spacy",""),
            "ft_50d_w2_CBOW_HS-w2v":os.path.join(path_embeddings, "ft_50d_w2_CBOW_HS.bin"),
            "ft_50d_w2_CBOW_HS-folder":os.path.join(path_embeddings, "ft_50d_w2_CBOW_HS-spacy",""),
            "ft_50d_w2_CBOW_NS-w2v":os.path.join(path_embeddings, "ft_50d_w2_CBOW_NS.bin"),
            "ft_50d_w2_CBOW_NS-folder":os.path.join(path_embeddings, "ft_50d_w2_CBOW_NS-spacy",""),
            "ft_50d_w2_SG_HS-w2v":os.path.join(path_embeddings, "ft_50d_w2_SG_HS.bin"),
            "ft_50d_w2_SG_HS-folder":os.path.join(path_embeddings, "ft_50d_w2_SG_HS-spacy",""),
            "ft_50d_w2_SG_NS-w2v":os.path.join(path_embeddings, "ft_50d_w2_SG_NS.bin"),
            "ft_50d_w2_SG_NS-folder":os.path.join(path_embeddings, "ft_50d_w2_SG_NS-spacy",""),
            "ft_50d_w3_SG_NS-w2v":os.path.join(path_embeddings, "ft_50d_w3_SG_NS.bin"),
            "ft_50d_w3_SG_NS-folder":os.path.join(path_embeddings, "ft_50d_w3_SG_NS-spacy",""),
            "ft_50d_w4_SG_NS-w2v":os.path.join(path_embeddings, "ft_50d_w4_SG_NS.bin"),
            "ft_50d_w4_SG_NS-folder":os.path.join(path_embeddings, "ft_50d_w4_SG_NS-spacy",""),
            "ft_50d_w5_SG_NS-w2v":os.path.join(path_embeddings, "ft_50d_w5_SG_NS.bin"),
            "ft_50d_w5_SG_NS-folder":os.path.join(path_embeddings, "ft_50d_w5_SG_NS-spacy",""),
            "ft_100d_w5_SG_NS-w2v":os.path.join(path_embeddings, "ft_100d_w5_SG_NS.bin"),
            "ft_100d_w5_SG_NS-folder":os.path.join(path_embeddings, "ft_100d_w5_SG_NS-spacy",""),
            "ft_150d_w5_SG_NS-w2v":os.path.join(path_embeddings, "ft_150d_w5_SG_NS.bin"),
            "ft_150d_w5_SG_NS-folder":os.path.join(path_embeddings, "ft_150d_w5_SG_NS-spacy","")            
    }
    
    pretrained_models = [
            "6d50",
            "6d100",
            "6d200",
            "6d300",
            "42d300",
            "840d300",
            "word2vec",
            "fasttext"
    ]
    
    lengths = {
            "6d50":50,
            "6d100":100,
            "6d200":200,
            "6d300":300,
            "42d300":300,
            "840d300":300,
            "word2vec":300,
            "fasttext": 300,
            "w2v_50d_w5_CBOW_HS":50,
            "w2v_50d_w5_CBOW_NS":50,
            "w2v_50d_w10_SG_HS":50,
            "w2v_50d_w10_SG_NS":50,
            "w2v_100d_w10_SG_NS":100,
            "w2v_100d_w10_SG_HS":100,
            "w2v_150d_w10_SG_NS":150,
            "w2v_300d_w10_SG_NS":300,
            "ft_50d_w2_CBOW_HS":50,
            "ft_50d_w2_CBOW_NS":50,
            "ft_50d_w2_SG_HS":50,
            "ft_50d_w2_SG_NS":50,
            "ft_50d_w3_SG_NS":50,
            "ft_50d_w4_SG_NS":50,
            "ft_50d_w5_SG_NS":50,
            "ft_100d_w5_SG_NS":100,
            "ft_150d_w5_SG_NS":150
    }
    
    models = {}
    
    nlp = spacy.load("en", vectors=False)
        
    def load_model(self, model, pretrained = True):
        """
        Loads a pre-trained word embedding to be used by this parser.
        
        Args:
            model (str): The model used. 
                One of pretrained {"6d50","6d100","6d200","6d300","42d300","840d300", "word2vec", "fasttext"}.
                One of the models trained on the abstracts' text data.
            
            pretrained(bool): Whether the used embeddings are pretrained or not.
        """
        pretrained = model in self.pretrained_models
        
        try:
            self.nlp = spacy.load(self.paths[model + "-folder"])
            self.length = self.lengths[model]
            
        except OSError:
            if not os.path.isfile(self.paths[model + "-w2v"]) and pretrained:
                from gensim.scripts.glove2word2vec import glove2word2vec
                print("Word2Vec format not present, generating it.")
                glove2word2vec(glove_input_file=self.paths[model], word2vec_output_file=self.paths[model + "-w2v"])
                        
            try:
                self.current_model = self.models[model + "-w2v"]
                self.length = self.lengths[model]
            except KeyError:
                if pretrained:
                    if model == "fasttext":
                        print("FastText not loaded yet, loading it.")
                        binary = False
                    elif model == "word2vec":
                        print("Word2Vec not loaded yet, loading it.")
                        binary = True
                    else:
                        print("Glove not loaded yet, loading it.")
                        binary = False
                else:
                    print("Model not loaded yet, loading it.")
                    binary = True
                
                self.models[model + "-w2v"] = KeyedVectors.load_word2vec_format(self.paths[model + "-w2v"], binary=binary)             
                self.current_model = self.models[model + "-w2v"]
                self.length = self.lengths[model]
            
            print("Setting up spacy vocab.")
            
            count = 0
            for word, o in self.current_model.vocab.items():
                count += 1
                self.nlp.vocab.set_vector(word, self.current_model[word])
            
            print("Done ({}). Saving to disk.".format(count))
            
            try:
                self.nlp.to_disk(self.paths[model + "-folder"])
            except FileNotFoundError:
                os.mkdir(self.paths[model + "-folder"])
                self.nlp.to_disk(self.paths[model + "-folder"])
    
    #################################################    
    def transform_matrix(self,sentence):
        """
        Transform a string into a matrix containing word embeddings as rows.
        
        Args:
            sentence (str): The string to be transformed. Can be a sentence or a whole document.
            
        Returns:
            A numpy matrix that contains word embeddings as rows of each word given by "sentence".
        """
        for w in self.nlp(sentence):
            try:
                m = np.concatenate((m,[w.vector]),axis=0)
            except NameError:
                m = np.array([w.vector])
            
        return m
    
    #################################################
    def transform_vector(self,sentence):
        """
        Transform a string into a vector containing concatenated word embeddings.
        
        Args:
            sentence (str): The string to be transformed. Can be a sentence or a whole document.
            
        Returns:
            A numpy array that contains concatenated word embeddings for each word given by "sentence".
        """
        
        for w in self.nlp(sentence):
            try:
                m = np.append(m,w.vector)
            except NameError:
                m = np.array(w.vector)
            
        return m
    
    #################################################
    def transform_vectors(self,sentences,batch_size=100):
        """
        Transform a list of strings into a list of vectors containing concatenated word embeddings.
        
        Args:
            sentence list(str): The list of strings to be transformed.
            
        Returns:
            A list of numpy arrays that contain concatenated word embeddings for each word given by "sentence".
        """
        
        vectors = list()
        for doc in self.nlp.tokenizer.pipe(sentences, batch_size=batch_size):
            m = np.empty(len(doc)*self.length)
            for i, w in enumerate(doc):
                m[(i*self.length):((i+1)*self.length)] = w.vector
            vectors.append(m)
            
        return vectors
    
    #################################################
    def transform_tensor_to_fixed_size(self,sentences,embeddings_size,spatial_size,batch_size=100):
        """
        Transform a list of strings into a tensor of fixed size containing word embeddings.
        
        Args:
            sentences list(str): The list of strings to be transformed.
            vector_size(int): 
            
        Returns:
            A list of numpy arrays that contain concatenated word embeddings for each word given by "sentence".
        """
        tensor = np.zeros(
                (len(sentences),embeddings_size,spatial_size),
                dtype=np.float32
        )
        
        for i_batch, doc in enumerate(self.nlp.tokenizer.pipe(sentences, batch_size=batch_size)):
            for i_word, word in enumerate(doc):
                if i_word >= spatial_size:
                    break
                tensor[i_batch,:,i_word] = word.vector
            
        return tensor
    
    #################################################
    def transform_avg_vectors(self,sentences,batch_size=100):
        """
        Transform a list of strings into a list of vectors containing averaged word embeddings.
        
        Args:
            sentence list(str): The list of strings to be transformed.
            
        Returns:
            A list of numpy arrays that contain averaged word embeddings for each word given by "sentence".
        """
        
        vectors = list()
        for doc in self.nlp.tokenizer.pipe(sentences, batch_size=batch_size):
            m = np.empty(self.length)
            for i, w in enumerate(doc):
                m = np.add(m,w.vector)
            vectors.append(m/len(doc))
            
        return vectors
       
      ################################################
    def transform_tfidf_avg_vectors(self, sentences, tfidf_weights, batch_size = 100):
        """
        Transform a list of strings into a list of vectors containing averaged
        weighted word embeddings.
        
        Args:
            sentence list(str): The list of strings to be transformed.
            
        Returns:
            A list of numpy arrays that contain averaged wieghted 
            word embeddings for each word given by "sentence".
        """
        vectors = list()
        max_weight = max(tfidf_weights.values())
        
        for doc in self.nlp.tokenizer.pipe(sentences, batch_size=batch_size):
            m = np.empty(self.length)
            sum_weights = 0
            for i, w in enumerate(doc):
                if w in tfidf_weights.keys():
                    weight = tfidf_weights[w]
                else:
                    weight = max_weight 
                sum_weights += weight
                m = np.add(m, np.multiply(w.vector, weight))
            vectors.append(m/sum_weights)
            
        return vectors
       
    #################################################
    def compute_tfidf_weights(self, sentences):
        """
        Calculate the TFIDF weights of a list of documents.
        
        Args:
            sentence list(str): The list of strings for which the TFIDF is to 
                                be computed.
            
        Returns:
            A dictionary (token, tfidf_score) containing the tfidf score 
            for each token in the "sentence".
        """
        tfidf = TfidfVectorizer(
                min_df = 0,
                max_df = 1.0,
                ) 
        tr_sentences = tfidf.fit_transform(sentences)
        feature_names = tfidf.get_feature_names()
        feature_index = tr_sentences.nonzero()[1]
        tfidf_scores = zip(feature_index, [tr_sentences[0, x] for x in feature_index])
        tfidf_weights = defaultdict(
                None,
                [(word, score) for word, score in [(feature_names[i], s) for (i, s) in tfidf_scores]]
                )
        
        return tfidf_weights
        
## Example:
#parser = EmbeddingsParser()
#parser.load_model("6d50")
#test = parser.transform_vector("This is a sentence.")
#print(test.shape)