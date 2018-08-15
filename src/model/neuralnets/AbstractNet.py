# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:31:45 2018

@author: Steff
"""

import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

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
    
    ##################################################
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
    
    ##################################################
    def rename(self,net_name_new):
        self.model_meta["net_name"] = net_name_new
        self.save_meta()
        
        path_save_states_new = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            AbstractNet.path_persistent,
            net_name_new
        )
        os.rename(self.path_save_states,path_save_states_new)
        
        self.path_save_states = path_save_states_new
        self.net_name = net_name_new
        
    ##################################################
    @staticmethod
    def load_meta(net_name):
        return torch.load(os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename_meta))
    
    ##################################################
    def save_meta(self):
        file_meta = os.path.join(
                AbstractNet.path_persistent,
                self.net_name,
                AbstractNet.filename + ".meta"
        )
        torch.save(self.model_meta,file_meta)
        
    ##################################################
    def save_state(self,epoch,final):
        file_state = os.path.join(
                AbstractNet.path_persistent,
                self.net_name,
                AbstractNet.filename + ("" if final else ".e"+str(epoch))
        )
        torch.save(self.model_state,file_state)
    
    ##################################################
    @staticmethod
    def save_state_exists(net_name):
        return os.path.isfile(
            os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename)
        ) & os.path.isfile(
            os.path.join(AbstractNet.path_persistent,net_name,AbstractNet.filename_meta)
        )
    
    ##################################################
    def plot_losses(self):
        ymax = max((max(self.model_state["losses"][0]),max(self.model_state["losses"][1])))
        #plt.rcParams["font.size"] = 20
        #plt.figure(figsize=(6,6))
        plt.plot(self.model_state["losses"][0])
        plt.plot(self.model_state["losses"][1])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend(["train","eval"],loc=3)
        plt.ylim(ymin=0,ymax=ymax+0.5)
        
        file_path = os.path.join(
            self.path_save_states,
            "losses.pdf"
        )
        plt.savefig(file_path, bbox_inches="tight")
        
        plt.show()
    
    ##################################################
    def print_stats(self):
        epchs = len(self.model_state["losses"][0])
        time_by_epoch = self.training_time/epchs
        epoch_min = np.argmin(self.model_state["losses"][1])
        
        file_path = os.path.join(
            self.path_save_states,
            "stats.txt"
        )
        with open(file_path,"w") as f:
            self._print("Total number of epochs trained: {}, avg. time per epoch: {}.".format(epchs,time_by_epoch),f)
            self._print("Total time trained: {}.".format(self.training_time),f)
            self._print("Lowest eval. loss at epoch {} = {}.".format(epoch_min,self.model_state["losses"][1][epoch_min]),f)
            
            f.write("\nLosses:\n")
            epoch_len = len(str(len(self.model_state["losses"][0])))
            for i, train_loss in enumerate(self.model_state["losses"][0]):
                eval_loss = self.model_state["losses"][1][i]
                f.write(("{:"+str(epoch_len)+"d}: {:13.10f} {:13.10f}\n").format(i,train_loss,eval_loss))
    
    ##################################################
    def _print(self,text,f):
        print(text)
        f.write(text + "\n")