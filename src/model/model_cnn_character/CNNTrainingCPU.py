# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:42:14 2018

@author: Steff
"""
import sys
import os
import gc

import torch
import torch.optim as optim

import math
sys.path.insert(0, os.path.join(os.getcwd(),"..","neuralnets"))

from TimerCounter import Timer

#### TRAINING PARAMETERS ####

USE_CUDA = False

# set a name which is used as directory for the save states in "data/processed/nn/<name>"
NET_NAME = "CNN-characters"

BATCH_SIZE = 500
EPOCHS = 1

DATA_TRAIN = "small"
SHUFFLE_TRAIN = True
DATA_TEST = "small"

MAX_DOCUMENT_SIZE = 1280

#WEIGHT_DECAY = 0.0005

VERBOSE_EPOCHS = True
VERBOSE_EPOCHS_STEPS = 10

# keep states of each epoch of the net
KEEP_STATES = True

# load the last saved state and continue training
CONTINUE_TRAINING = True

########### CUDA ############

if USE_CUDA:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.fastest = True

#############################

from BatchifiedCharactersDataLazy import BatchifiedCharactersDataLazy
from CharacterCNNet import CharacterCNNet

print(">>> loading data")

timer = Timer()
d_train = BatchifiedCharactersDataLazy(
        use_cuda=USE_CUDA,
        data_which=DATA_TRAIN,
        max_document_size=MAX_DOCUMENT_SIZE
)
gc.collect()
d_test = BatchifiedCharactersDataLazy(
        use_cuda=USE_CUDA,
        training=False,
        classes=d_train.classes,
        data_which=DATA_TEST,
        max_document_size=MAX_DOCUMENT_SIZE
)
gc.collect()

print(">>> creating net")

# create Net
net = CharacterCNNet(
        NET_NAME,
        classes=d_train.classes,
        letters_size=d_train.letters_size
)
if USE_CUDA:
    net.cuda()

# create optimizer
#optimizer = optim.SGD(net.parameters(), lr=1)
optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)
#optimizer = optim.Adadelta(
#        net.parameters()
#        ,weight_decay=WEIGHT_DECAY
#)

losses_train = []
losses_test = []
epoch = 0

if CONTINUE_TRAINING and CharacterCNNet.save_state_exists(NET_NAME):
    model_state = net.load_state(optimizer)
    epoch = model_state["epoch"] + 1
    losses_train, losses_test = model_state["losses"]
    print("Continue training at epoch: {}".format(epoch))

#for param in net.conv1.parameters():
#    print(param)

print(">>> starting training")

# batch data loading
for epoch in range(epoch,epoch+EPOCHS):
    print("============ EPOCH {} ============".format(epoch))
    
    ### TRAINING
    net.train()
    
    running_loss = 0
    divisor = 0
    timer.tic()
    if VERBOSE_EPOCHS:
        timer.set_counter(d_train.data_size,max=VERBOSE_EPOCHS_STEPS)
    
    if VERBOSE_EPOCHS:
        print("Batchify.")
    d_train.batchify(BATCH_SIZE,shuffle=SHUFFLE_TRAIN)
    while d_train.has_next_batch():
        inputs, labels = d_train.next_batch()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.loss(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if VERBOSE_EPOCHS:
            timer.count(len(inputs))
        
    running_loss = running_loss/math.ceil(d_train.batch_total)
    print("Train ==> Epoch: {}, Loss: {}".format(epoch,running_loss))
    net.training_time += timer.toc()
    losses_train.append(running_loss)
    
    del outputs
    del loss
    del inputs
    del labels
    gc.collect()
    
    ### EVALUATION
    net.eval()
    
    running_loss = 0
    timer.tic()
    if VERBOSE_EPOCHS:
        timer.set_counter(d_test.data_size,max=VERBOSE_EPOCHS_STEPS)
    
    d_test.batchify(BATCH_SIZE,shuffle=False)
    while d_test.has_next_batch():
        inputs, labels = d_test.next_batch()
        
        with torch.no_grad():
            outputs = net(inputs)
            loss = net.loss(outputs,labels)
        
        running_loss += loss.item()
        if VERBOSE_EPOCHS:
            timer.count(len(inputs))
        
    running_loss = running_loss/math.ceil(d_test.batch_total)
    print("Eval ==> Epoch: {}, Loss: {}".format(epoch,running_loss))
    timer.toc()
    losses_test.append(running_loss)
    
    if KEEP_STATES:
        net.save_state(epoch,[losses_train,losses_test],optimizer)

#save final model state
net.save_state(epoch,[losses_train,losses_test],optimizer,True)

# draw losses
net.plot_losses()
net.print_stats()

#for param in net.conv1.parameters():
#    print(param)
#    print(param.grad.data.sum())