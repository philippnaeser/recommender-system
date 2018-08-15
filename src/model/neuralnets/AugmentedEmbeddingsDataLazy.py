# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 01:14:57 2018

@author: Steff
"""
import os
import pickle
import sys
import math

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
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
    
    ##################################################
    def __del__(self):
        pass
    
    ##################################################
    def __init__(self,use_cuda=True,data_which="small",embeddings_model="6d50"):
        self.data_labels = list()
        self.data_abstracts = list()
        
        data_type = "training"
        
        self.filepath = os.path.join(self.path_persistent,
                                     "-".join([data_type,
                                              "data-cnn-augmented",
                                              data_which
                                              ])
                                     )
        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        print("Loading word embeddings and parser.")
        self.embeddings_parser = EmbeddingsParser()
        self.embeddings_parser.load_model(embeddings_model)
        self.embeddings_size = EmbeddingsParser.lengths[embeddings_model]
    
        if os.path.isdir(self.filepath):
            print("Loading {} data from disk.".format(data_type))
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"rb") as f:
                self.classes, self.data_size = pickle.load(f)
                
            for i in range(len(self.classes)-1):
                self._load_file(i)
        
        else:
            print("{} data not on disk.".format(data_type))
            os.mkdir(self.filepath)
            print("Loading and preprocessing {} data.".format(data_type))
            
            print("Loading SciGraph.")
            self.d = SciGraphLoader()
            self.d.training_data_for_abstracts(data_which)
                    
            # initialize labels
            self.l = LabelEncoder()
            self.d.data["conferenceseries"] = self.l.fit_transform(self.d.data["conferenceseries"])
            self.classes = np.array(self.l.classes_)
            # append a class for conferences not in the training data
            self.classes = np.append(
                    self.classes,
                    None
            )

            print("Preprocessing abstracts.")
            self.d.data["chapter_abstract"] = self.d.data["chapter_abstract"].str.lower()
            
            print("Analyzing abstracts.")
            
            timer = Timer()
            
            abstracts = {}
            for index, series in self.d.data.iterrows():
                try:
                    abstracts[series.conferenceseries]
                except KeyError:
                    abstracts[series.conferenceseries] = list()
                abstracts[series.conferenceseries].append(series.chapter_abstract)
            
            print("Augmenting abstracts.")
            
            conference_list = []
            abstract_list = []
            
            self.data_size = 0
            
            timer.set_counter(len(self.d.data),max=5)
            # Loop through all unique conferenceseries.
            for conf in abstracts:
                a = abstracts[conf]
                multiplier = math.ceil(1000/len(a))
                # Loop through all abstracts of the current conferenceseries
                # and generate additional abstracts.
                for abstract in a:
                    aug = AugmentedAbstract(
                            abstract,
                            conf=conf,
                            multiplier=multiplier,
                            conference_list=conference_list,
                            abstract_list=abstract_list,
                            translate=False
                    )
                    self.data_size += aug.length
                    del aug
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

        # get labels.
        labels = self.data_labels[i_from:i_to]
        labels = torch.tensor(
                labels,
                device=self.device,
                dtype=torch.long
        ).view(len(labels))

        # get abstracts.
        batch = self.embeddings_parser.transform_tensor_to_fixed_size(
                self.data_abstracts[i_from:i_to],
                embeddings_size=self.embeddings_size,
                spatial_size=300
        )
        batch = torch.tensor(
                batch,
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
            if len(data_abstracts) != len(data_labels):
                print("whats going on???? {}".format(conf))



class AugmentedAbstract():
    
    # Languages which are used for translation augmentation.
    # = 5 possible additional abstracts.
    languages = ["de","es","fr","it","pt"]
    
    ##################################################
    def __init__(self,text,conf,multiplier,abstract_list,conference_list,translate=True,shuffle=True):
        # List from outside to add all abstracts to.
        self.abstracts = abstract_list
        # Add the original abstract.
        self.abstracts.append(text)
        # List from outside to add all labels to.
        self.conferences = conference_list
        # Add label for the original abstract.
        self.conferences.append(conf)
        self.conf = conf
        
        self.blob = TextBlob(text)
        self.length = 1
        self.multiplier = multiplier
        
        if translate:
            # Translation takes long, because they work via API calls to Google translate.
            self._generate_translations()
        if shuffle:
            # Shuffling is fast / done offline.
            self._generate_shuffles()
    
    ##################################################
    def _generate_translations(self):
        for lang in AugmentedAbstract.languages:
            if self.length >= self.multiplier:
                break
            # Translate the abstract back and forth, which changes it a bit.
            translated = self.blob.translate(from_lang="en", to=lang)
            translated = translated.translate(from_lang=lang, to="en")
            self.abstracts.append(translated.raw)
            self.conferences.append(self.conf)
            self.length += 1
    
    ##################################################
    def _generate_shuffles(self):
        # Generate a list with all sentences of the abstract.
        sentences = list()
        for s in self.blob.sentences:
            sentences.append(s.raw)
        
        # Add shuffled abstracts until the wanted multiplier is reached.
        for i, p in enumerate(multiset_permutations(sentences)):
            if i > 0:
                if self.length >= self.multiplier:
                    break
                self.abstracts.append(" ".join(p))
                self.conferences.append(self.conf)
                self.length += 1