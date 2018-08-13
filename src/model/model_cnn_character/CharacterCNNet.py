# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:42:14 2018

@author: Steff
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from AbstractNet import AbstractNet
sys.path.insert(0, os.path.join(
        os.path.realpath(__file__),"..","..","neuralnets")
)

class CharacterCNNet(AbstractNet):
    
    ##################################################
    @classmethod
    def from_disk(self,net_name,optimizer=None,epoch=None):
        """
            Initialize by loading a save state.
        """
        if not CharacterCNNet.save_state_exists(net_name):
            raise FileNotFoundError("Save state for model '{}' not found.".format(net_name))
            
        meta = CharacterCNNet.load_meta(net_name)
        net = CharacterCNNet(
                net_name=meta["net_name"],
                classes=meta["classes"],
                letters_size=meta["letters_size"]
        )
        net.load_state(optimizer,epoch)
        
        return net
    
    ##################################################
    def __init__(self,net_name,classes,letters_size):
        super(CharacterCNNet, self).__init__(net_name)

        self.net_name = net_name

        self.classes = classes
        self.num_classes = len(classes)
        self.letters_size = letters_size
        
        # 1 channel in, <filters> channels out
        self.conv1 = nn.Conv1d(
                in_channels=letters_size,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
        )
        
        self.conv2 = ConvBlock(64,64)
        self.conv3 = ConvBlock(64,128)
        self.conv4 = ConvBlock(128,256)
        self.conv5 = ConvBlock(256,512)
        
        self.fc1 = nn.Linear(4096,2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048,2048)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2048,self.num_classes)
        
        self.loss = nn.CrossEntropyLoss()
    
    ##################################################
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool1d(
                x,
                kernel_size=3,
                stride=2,
                padding=1
        )
        x = self.conv3(x)
        x = F.max_pool1d(
                x,
                kernel_size=3,
                stride=2,
                padding=1
        )
        x = self.conv4(x)
        x = F.max_pool1d(
                x,
                kernel_size=3,
                stride=2,
                padding=1
        )
        x = self.conv5(x)
        
        x = self.kmax_pooling(x,2,8) # [B,512,8]
        x = x.view(x.size()[0],-1)   # [B,4096]
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x#self.softmax(x)
    
    ##################################################
    def kmax_pooling(self,x,dim,k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)
    
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
                "classes":self.classes,
                "net_name":self.net_name,
                "letters_size":self.letters_size
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
        
        
        
class ConvBlock(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1
        )
        self.norm1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1
        )
        self.norm2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        return x