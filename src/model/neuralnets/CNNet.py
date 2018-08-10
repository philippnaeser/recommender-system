# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from AbstractNet import AbstractNet

class CNNet(AbstractNet):
    
    ##################################################
    @classmethod
    def from_disk(self,net_name,optimizer=None,epoch=None):
        """
            Initialize by loading a save state.
        """
        if not CNNet.save_state_exists(net_name):
            raise FileNotFoundError("Save state for model '{}' not found.".format(net_name))
            
        meta = CNNet.load_meta(net_name)
        net = CNNet(
                net_name=meta["net_name"],
                embedding_model=meta["embedding_model"],
                embedding_size=meta["embedding_size"],
                classes=meta["classes"],
                filters=meta["filters"]
        )
        net.load_state(optimizer,epoch)
        
        return net
    
    ##################################################
    def __init__(self,net_name,embedding_model,embedding_size,classes,filters=100):
        super(CNNet, self).__init__(net_name)
        
        self.filters = filters
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        self.classes = classes
        self.num_classes = len(classes)
        self.net_name = net_name
        
        # 1 channel in, <filters> channels out
        self.conv1 = nn.Conv1d(1,filters,
                               5*self.embedding_size,
                               stride=1*self.embedding_size
        )
        self.conv2 = nn.Conv1d(1,filters,
                               4*self.embedding_size,
                               stride=1*self.embedding_size
        )
        self.conv3 = nn.Conv1d(1,filters,
                               3*self.embedding_size,
                               stride=1*self.embedding_size
        )
        #self.fc1 = nn.Linear(100,100)
        self.fc2 = nn.Linear(3*filters,self.num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
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
        
        #x = self.fc1(x.view(x.size()[0],-1))
        #x = F.relu(x)
        #x = self.fc2(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x.view(x.size()[0],-1))
        
        return x#self.softmax(x)
    
    ##################################################
    def save_state(self,epoch,losses,optimizer,final=False):
        self.model_state = {
                "epoch":epoch,
                "losses":losses,
                "model":self.state_dict(),
                "optimizer":optimizer.state_dict(),
                "training_time":self.training_time
        }
        file_state = os.path.join(
                AbstractNet.path_persistent,
                self.net_name,
                AbstractNet.filename + ("" if final else ".e"+str(epoch))
        )
        
        self.model_meta = {
                "embedding_size":self.embedding_size,
                "embedding_model":self.embedding_model,
                "classes":self.classes,
                "filters":self.filters,
                "net_name":self.net_name
        }
        file_meta = os.path.join(
                AbstractNet.path_persistent,
                self.net_name,
                AbstractNet.filename + ".meta"
        )

        torch.save(self.model_state, file_state)
        torch.save(self.model_meta, file_meta)
    
    ##################################################
    def load_state(self,optimizer=None,epoch=None):
        self.model_state = torch.load(os.path.join(
                AbstractNet.path_persistent,
                self.net_name,
                AbstractNet.filename + ("" if epoch is None else ".e"+str(epoch))
        ))
        
        self.load_state_dict(self.model_state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(self.model_state["optimizer"])
        if "training_time" in self.model_state:
            self.training_time = self.model_state["training_time"]
        
        return self.model_state
    
    ##################################################
    @staticmethod
    def load_meta(net_name):
        return torch.load(os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename_meta))
    
    ##################################################
    @staticmethod
    def save_state_exists(net_name):
        return os.path.isfile(
            os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename)
        ) & os.path.isfile(
            os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename_meta)
        )