# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from AbstractNet import AbstractNet

class RNNet(AbstractNet):
    
    def __init__(self,net_name,letters_size,num_classes,hidden_size=100):
        super(RNNet, self).__init__(net_name)
        
        self.lstm = nn.LSTM(
                input_size=letters_size
                ,hidden_size=hidden_size
                #,num_layers=1 # default: 1
                #,batch_first=False # default: False
                #,dropout=0 # default: 0
                #,bidirectional=False # default: False
        )
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(100,num_classes)
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x, hidden = self.lstm(x)
        
        x = self.dropout(x)
        
        x = x[x.size()[0]-1,:,:].squeeze(0)
        
        x = self.fc(x)
        
        return x#self.softmax(x)