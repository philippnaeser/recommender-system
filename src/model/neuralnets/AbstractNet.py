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
    filename_meta = "model-save-state.meta"
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "processed",
            "nn"
        )
    
    def __init__(self,net_name):
        super(AbstractNet, self).__init__()
        
        self.path_save_states = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            AbstractNet.path_persistent,
            net_name
        )
        
        #create path for save states
        if not os.path.isdir(self.path_save_states):
            os.mkdir(self.path_save_states)
            
        self.training_time = 0
    
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