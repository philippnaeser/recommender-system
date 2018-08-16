# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 01:14:57 2018

@author: Steff

This class provides training data for neural network training, which gets augmented by
shuffling or translating abstracts.
TextBlob is used to break abstracts into sentences and for translation.
multiset_permutations provides deterministic permutations (in this context referred
to as "shuffling") of an abstract
All abstracts are cached to disk as lists of tokens, this saves tokenization during
training time.
The SpaCy tokenizer is used to create the tokens.
During training, a gensim KeyedVector is used to access each token embedding.

Usage:
    data = AugmentedEmbeddingsDataLazy(
            use_cuda=False,
            data_which="small",
            embeddings_model="w2v_100d_w10_SG_NS"
    )
    data.batchify(1000)
    input_tensors, label_tensor = data.next_batch()

"""
import os
import pickle
import sys
import math

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import spacy
from textblob import TextBlob
from sympy.utilities.iterables import multiset_permutations

sys.path.insert(0, os.path.join(os.getcwd(),"..","..","data"))
from EmbeddingsParser import EmbeddingsParser
from DataLoader import DataLoader as SciGraphLoader
from TimerCounter import Timer

class AugmentedEmbeddingsDataLazy():
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "interim",
            "neuralnet_training"
    )
    
    # Languages which are used for translation augmentation.
    # = 5 possible additional abstracts.
    languages = ["de","es","fr","it","pt"]
    
    ##################################################
    def __del__(self):
        pass
    
    ##################################################
    def __init__(self, use_cuda=True, data_which="small", embeddings_model="6d50", spatial_size=300):
        """
        
        Args:
            use_cuda: Determines if tensor should be created on CPU or GPU.
            data_which: Determines the size of the training data used to augment.
            embeddings_model: Determines which word embeddings to use.
            spatial_size: Sets the maximum length of an abstract in number of tokens.
        """
        data_type = "training"
        self.use_cuda = use_cuda
        self.embeddings_model = embeddings_model
        self.spatial_size = spatial_size
        # Storage for all abstracts and labels. Abstracts are transformed into tensors during training.
        self.data_labels = list()
        self.data_abstracts = list()
        
        # Location of cached files.
        self.filepath = os.path.join(self.path_persistent,
                                     "-".join([data_type,
                                              "data-cnn-augmented-tokenized",
                                              data_which
                                              ])
                                     )
        if use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        print("Loading word embeddings and parser.")
        self.tokenizer = spacy.load("en", vectors=False, disable=['tagger','ner','textcat']).tokenizer
        self.embedding_vectors = EmbeddingsParser._get_keyed_vectors(embeddings_model,True)
        self.embeddings_size = EmbeddingsParser.lengths[embeddings_model]
    
        # Check if the data is already present on disk.
        if os.path.isdir(self.filepath):
            print("Loading {} data from disk.".format(data_type))
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"rb") as f:
                self.classes, self.data_size = pickle.load(f)
                
            # Load all abstracts into memory.
            for i in range(len(self.classes)-1):
                self._load_file(i)
        
        else:
            timer = Timer()
            print("Training data not on disk.")
            os.mkdir(self.filepath)
            print("Loading and preprocessing Training data.")
            
            print("Loading SciGraph.")
            self.d = SciGraphLoader()
            self.d.training_data_for_abstracts(data_which)
                    
            # Initialize labels.
            self.l = LabelEncoder()
            self.d.data["conferenceseries"] = self.l.fit_transform(self.d.data["conferenceseries"])
            self.classes = np.array(self.l.classes_)
            # Append a class for conferences not in the training data.
            self.classes = np.append(
                    self.classes,
                    None
            )

            print("Preprocessing abstracts.")
            self.d.data["chapter_abstract"] = self.d.data["chapter_abstract"].str.lower()
            
            print("Analyzing abstracts.")
            
            # Create a dict containing abstracts by conferenceseries.
            abstracts = {}
            for index, series in self.d.data.iterrows():
                try:
                    abstracts[series.conferenceseries]
                except KeyError:
                    abstracts[series.conferenceseries] = list()
                abstracts[series.conferenceseries].append(series.chapter_abstract)
            
            print("Augmenting abstracts.")
            # Lists which are filled by _augment_abstract() and saved to disk.
            conference_list = []
            abstract_list = []
            
            # Cound the total number of rows.
            self.data_size = 0
            
            timer.set_counter(len(self.d.data),max=100)
            # Loop through all unique conferenceseries.
            for conf in abstracts:
                a = abstracts[conf]
                multiplier = math.ceil(1000/len(a))
                # Loop through all abstracts of the current conferenceseries
                # and generate additional abstracts.
                for abstract in a:
                    self._augment_abstract(
                            abstract,
                            conf=conf,
                            multiplier=multiplier,
                            conference_list=conference_list,
                            abstract_list=abstract_list,
                            translate=False
                    )
                    timer.count()
                # Save abstracts of current conferenceseries into a pickle file.
                file = os.path.join(self.filepath,"".join(["abstracts.",str(conf),".pkl"]))
                with open(file,"wb") as f:
                    pickle.dump([abstract_list, conference_list], f)
                # Extend global lists.
                self.data_abstracts.extend(abstract_list)
                self.data_labels.extend(conference_list)
                # Delete local lists and create new ones.
                del conference_list
                del abstract_list
                conference_list = []
                abstract_list = []
                
            print("Saving meta.")
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"wb") as f:
                pickle.dump([self.classes, self.data_size], f)

            del self.d
            del self.l
        
        # Turn data into numpy arrays.
        self.data_abstracts = np.array(self.data_abstracts,dtype=object)
        self.data_labels = np.array(self.data_labels,dtype=np.int32)
    
    ##################################################
    def batchify(self,batch_size,shuffle=True):
        """
        Batchifies the dataset.
        
        Args:
            size (int): number of rows in a batch.
            num_chunks (int): number of chunks to preload into memory.
            shuffle (bool): retrieve randomized batches
        """
        self.batch_size = batch_size
        self.batches = np.arange(self.data_size,step=batch_size)
        
        self.batch_current = 0
        self.batch_max = len(self.batches)
        
        if shuffle:
            order = np.arange(len(self.data_abstracts))
            np.random.shuffle(order)
            self.data_abstracts = self.data_abstracts[order]
            self.data_labels = self.data_labels[order]

    ##################################################
    def next_batch(self):            
        i_from = self.batch_current*self.batch_size
        i_to = i_from + self.batch_size
        
        # Get and transform labels.
        labels = self.data_labels[i_from:i_to]
        labels = torch.tensor(
                labels,
                device=self.device,
                dtype=torch.long
        ).view(len(labels))

        # Get and transform abstracts.
        spatial_size = self.spatial_size
        abstracts = self.data_abstracts[i_from:i_to]

        # Create a zeros-numpy tensor and fill it where necessary.
        tensor = np.zeros(
                (len(abstracts),self.embeddings_size,spatial_size),
                dtype=np.float32
        )
        for i_batch, doc in enumerate(abstracts):
            for i_word, word in enumerate(doc):
                if i_word >= spatial_size:
                    break
                try:
                    tensor[i_batch,:,i_word] = self.embedding_vectors[word]
                except KeyError:
                    pass
 
        batch = torch.tensor(
                tensor,
                device=self.device,
                dtype=torch.float32
        )
        
        self.batch_current += 1
        
        return batch, labels
    
    ##################################################
    def has_next_batch(self):
        return self.batch_current < self.batch_max
        
    ##################################################
    def num_classes(self):
        return len(self.classes)
    
    ##################################################
    def _load_file(self,conf):
        file = os.path.join(self.filepath,"".join(["abstracts.",str(conf),".pkl"]))
        with open(file,"rb") as f:
            data_abstracts, data_labels = pickle.load(f)
            self.data_abstracts.extend(data_abstracts)
            self.data_labels.extend(data_labels)

    ##################################################
    def _augment_abstract(self,text,conf,multiplier,abstract_list,conference_list,translate=True,shuffle=True):
        # Add the original abstract.
        abstract_list.append(text)
        # Add label for the original abstract.
        conference_list.append(conf)
        
        blob = TextBlob(text)
        length = 1
        self.data_size += 1
        
        if translate:
            # Translation takes long, because they work via API calls to Google translate.
            for lang in self.languages:
                if length >= multiplier:
                    break
                # Translate the abstract back and forth, which changes it a bit.
                translated = blob.translate(from_lang="en", to=lang)
                translated = translated.translate(from_lang=lang, to="en")
                abstract_list.append(translated.raw)
                conference_list.append(conf)
                length += 1
                self.data_size += 1
        if shuffle:
            # Shuffling is fast / done offline.
            # Generate a list with all sentences of the abstract.
            sentences = list()
            for s in blob.sentences:
                sentences.append(s.raw)
            
            # Add shuffled abstracts until the wanted multiplier is reached.
            for i, p in enumerate(multiset_permutations(sentences)):
                if i > 0:
                    if length >= multiplier:
                        break
                    abstract_new = " ".join(p)
                    abstract_tokenized = self.tokenizer(abstract_new)
                    abstract_tokenized_list = [t.text for t in abstract_tokenized]
                    abstract_list.append(abstract_tokenized_list)
                    conference_list.append(conf)
                    length += 1
                    self.data_size += 1
    


#timer = Timer()
#test = AugmentedEmbeddingsDataLazy(use_cuda=False,data_which="small",embeddings_model="w2v_100d_w10_SG_NS")
#test.batchify(1000)

#timer.tic()
#while test.has_next_batch():
#    test.next_batch()
#timer.toc()