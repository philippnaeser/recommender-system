# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neuralnets"))
from AbstractNet import AbstractNet
sys.path.insert(0, os.path.join(
        os.path.dirname(__file__),"..","..","data")
)
from EmbeddingsParser import EmbeddingsParser

class CNNet2(AbstractNet):
    
    ##################################################
    @classmethod
    def from_disk(self,net_name,optimizer=None,epoch=None):
        """
            Initialize by loading a save state.
        """
        if not CNNet2.save_state_exists(net_name):
            raise FileNotFoundError("Save state for model '{}' not found.".format(net_name))
            
        meta = CNNet2.load_meta(net_name)
        net = CNNet2(
                net_name=meta["net_name"],
                embedding_model=meta["embedding_model"],
                classes=meta["classes"],
                filters=meta["filters"]
        )
        net.load_state(optimizer,epoch)
        net.meta = meta
        
        return net
    
    ##################################################
    def __init__(self,net_name,embedding_model,classes,filters=100):
        super(CNNet2, self).__init__(net_name)
        
        self.filters = filters
        self.embedding_model = embedding_model
        self.embedding_size = EmbeddingsParser.lengths[embedding_model]
        self.classes = classes
        self.num_classes = len(classes)
        self.net_name = net_name
        
        # 1 channel in, <filters> channels out
        self.conv1 = nn.Conv1d(
                in_channels=self.embedding_size,
                out_channels=filters,
                kernel_size=5,
                stride=1
        )
        self.conv2 = nn.Conv1d(
                in_channels=self.embedding_size,
                out_channels=filters,
                kernel_size=4,
                stride=1
        )
        self.conv3 = nn.Conv1d(
                in_channels=self.embedding_size,
                out_channels=filters,
                kernel_size=3,
                stride=1
        )
        
        self.fc1 = nn.Linear(3*filters,1024)
        self.fc2 = nn.Linear(1024,self.num_classes)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        
        #self.softmax = nn.LogSoftmax(dim=1)
        
        self.loss = nn.CrossEntropyLoss()
    
    ##################################################
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = F.max_pool1d(x1, kernel_size=x1.size()[2])
        
        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = F.max_pool1d(x2, kernel_size=x2.size()[2])
        
        x3 = self.conv3(x)
        x3 = F.relu(x3)
        x3 = F.max_pool1d(x3, kernel_size=x3.size()[2])
        
        x = torch.cat((x1,x2,x3),dim=1)
        
        x = self.dropout1(x)
        
        x = self.fc1(x.view(x.size()[0],-1))
        x = F.relu(x)
        
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        return x#self.softmax(x)
    
    ##################################################
    def save(self,epoch,losses,optimizer,final=False):
        self.model_state = {
                "epoch":epoch,
                "losses":losses,
                "model":self.state_dict(),
                "optimizer":optimizer.state_dict(),
                "training_time":self.training_time
        }
        self.save_state(epoch,final)
        
        self.model_meta = {
                "embedding_size":self.embedding_size,
                "embedding_model":self.embedding_model,
                "classes":self.classes,
                "filters":self.filters,
                "net_name":self.net_name
        }
        self.save_meta()
    
    ##################################################
    def load_state(self,optimizer=None,epoch=None):
        if torch.cuda.is_available():
            self.model_state = torch.load(os.path.join(
                    AbstractNet.path_persistent,
                    self.net_name,
                    AbstractNet.filename + ("" if epoch is None else ".e"+str(epoch))
            ))
        else:
            self.model_state = torch.load(
                os.path.join(
                    AbstractNet.path_persistent,
                    self.net_name,
                    AbstractNet.filename + ("" if epoch is None else ".e"+str(epoch))
                ),
                map_location="cpu"
            )
        
        self.load_state_dict(self.model_state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(self.model_state["optimizer"])
        if "training_time" in self.model_state:
            self.training_time = self.model_state["training_time"]
        
        return self.model_state