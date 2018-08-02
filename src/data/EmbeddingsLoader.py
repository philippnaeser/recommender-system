# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 23:38:49 2018

@author: Andreea
"""

import os
from gensim.models import Word2Vec, FastText
import numpy as np

class EmbeddingsLoader():

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
            "ft_150d_w5_SG_NS-folder":os.path.join(path_embeddings, "ft_150d_w5_SG_NS-spacy",""), 
            "w2v_100d_w10_SG_NS":os.path.join(path_embeddings, "w2v_100d_w10_SG_NS.model")
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
            
    def load_model(self, model, pretrained = True):
        """
        Loads a pre-trained word embedding to be used by this parser.
        
        Args:
            model (str): The model used. 
                One of pretrained {"6d50","6d100","6d200","6d300","42d300","840d300", "word2vec", "fasttext"}.
                One of the models trained on the abstracts' text data.
            
            pretrained(bool): Whether the used embeddings are pretrained or not.
            
        Returns:
            w2v (dict): Dictionary of (word, feature_vector)
        """
        pretrained = model in self.pretrained_models
        
        if pretrained:
            with open(self.paths[model], "rb") as lines:
                self.w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) 
                            for line in lines}
                
        else:
            model = Word2Vec.load(self.paths[model]) 
            self.w2v = dict(zip(model.wv.vocab, model.wv.vectors))

        return self.w2v
    
    def get_dimension(self, model):
        """
        Return the dimension of the feature vectors.
        
        Args: 
            model (str): The model used. 
                One of pretrained {"6d50","6d100","6d200","6d300","42d300","840d300", "word2vec", "fasttext"}.
                One of the models trained on the abstracts' text data.
                
        Returns:
            length (int): The dimension of the feature vector.
        """
        try:
            length = self.lengths[model]
        except KeyError:
            print("The {} model does not exist.".format(model))
            length = None
            
        return length
    
## Example:
#loader = EmbeddingsLoader()
#embedding_model = "w2v_100d_w10_SG_NS"
#w2v = loader.load_model(embedding_model)
#dimension = loader.get_dimension(embedding_model)
