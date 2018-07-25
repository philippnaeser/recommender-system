# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from AbstractNet import AbstractNet

class CNNet(AbstractNet):
    
    def __init__(self,net_name,embedding_size,classes,filters=100):
        super(CNNet, self).__init__(net_name)
        
        self.embedding_size = embedding_size
        
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
        self.fc2 = nn.Linear(3*filters,classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.loss = nn.CrossEntropyLoss()
        
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