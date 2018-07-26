# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""

import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class AbstractNet(nn.Module):
    
    filename = "model-save-state"
    
    def __init__(self,net_name):
        super(AbstractNet, self).__init__()
        
        self.path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "processed",
            "nn",
            net_name
        )
        
        #create path for save states
        if not os.path.isdir(self.path_persistent):
            os.mkdir(self.path_persistent)
            
        self.training_time = 0
    
    def save_state(self,epoch,losses,optimizer,final=False):
        self.model_state = {
                    "epoch":epoch,
                    "losses":losses,
                    "model":self.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "training_time":self.training_time
        }
        file = os.path.join(
                self.path_persistent,
                self.filename + ("" if final else ".e"+str(epoch))
        )
        torch.save(self.model_state, file)
        
    def load_state(self,optimizer=None):
        self.model_state = torch.load(os.path.join(self.path_persistent,self.filename))
        
        self.load_state_dict(self.model_state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(self.model_state["optimizer"])
        if "training_time" in self.model_state:
            self.training_time = self.model_state["training_time"]
        
        return self.model_state["epoch"], self.model_state["losses"]
    
    def plot_losses(self):
        ymax = max((max(self.model_state["losses"][0]),max(self.model_state["losses"][1])))
        plt.plot(self.model_state["losses"][0])
        plt.plot(self.model_state["losses"][1])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend(["train","eval"],loc=3)
        plt.ylim(ymin=0,ymax=ymax+0.5)
        plt.show()
        
    def print_stats(self):
        epchs = len(self.model_state["losses"][0])
        time_by_epoch = self.training_time/epchs
        print("Total number of epochs trained: {}, avg. time per epoch: {}.".format(epchs,time_by_epoch))
        print("Total time trained: {}.".format(self.training_time))
        
    def save_state_exists(self):
        return os.path.isfile(
            os.path.join(self.path_persistent,self.filename)
        )