import numpy as np
import pandas as pd
import gc
import pickle
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import os
import sys
import time
import math
sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))
from EmbeddingsParser import EmbeddingsParser
from DataLoader import DataLoader as SciGraphLoader

#### CUDA

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

#### TRAINING PARAMETERS

BATCH_SIZE = 500
CHUNKS_IN_MEMORY = 10
CHUNK_SIZE = 60000
EPOCHS = 50
DATA_TRAIN = "small"
SHUFFLE_TRAIN = False
DATA_TEST = "small"
EMBEDDING_MODEL = "6d50"
EMBEDDING_SIZE = 50
NUM_FILTERS_CONV = 50
WEIGHT_DECAY = 0.0005

VERBOSE_EPOCHS = False
VERBOSE_EPOCHS_STEPS = 10

####

class Timer:
    start_time = []
    
    ### start runtime check
    def tic(self):
        self.start_time.append(time.time())
    
    ### print runtime information
    def toc(self):
        diff = (time.time() - self.start_time.pop())
        print("Timer :: toc --- %s seconds ---" % diff)
        return diff
        
    def set_counter(self,c,max=100):
        self.counter_max = c
        self.counter = 0
        self.checkpoint = int(self.counter_max/max)
        self.step = self.checkpoint
        self.tic()
        
    def count(self,add=1):
        self.counter = self.counter + add
        
        if (self.counter >= self.checkpoint):
            print("Timer :: Checkpoint reached: {}%".format(int(self.counter*100/self.counter_max)))
            self.toc()
            self.checkpoint += self.step
            if self.checkpoint <= self.counter_max:
                self.tic()


class BatchifiedData(Dataset):
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "interim"
    )
    
    def __del__(self):
        pass
    
    def __init__(self,training=True,data_which="small",glove_model="6d50",classes=None,chunk_size=1000):
        if training:
            data_type = "training"
        else:
            data_type = "test"
        
        self.filepath = os.path.join(self.path_persistent,data_type+"-data-cnn-"+data_which+glove_model)
        timer = Timer()
        
        if os.path.isdir(self.filepath):
            print("Loading {} data from disk. Lazy loading inputs.".format(data_type))
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"rb") as f:
                self.classes = pickle.load(f)
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"rb") as f:
                self.chunks_max, self.chunk_size, self.size = pickle.load(f)
        
        else:
            print("{} data not on disk.".format(data_type))
            os.mkdir(self.filepath)
            print("Loading and preprocessing {} data.".format(data_type))
            
            timer.tic()
            print("Loading SciGraph.")
            self.d = SciGraphLoader()
            if training:
                self.d.training_data(data_which).abstracts()
            else:
                self.d.test_data(data_which).abstracts()
            timer.toc()
            
            # drop empty abstracts
            self.d.data.drop(
                list(self.d.data[pd.isnull(self.d.data.chapter_abstract)].index),
                inplace=True
            )
            self.d.data.reset_index(inplace=True)
                    
            # initialize labels
            if training:
                self.l = LabelEncoder()
                self.d.data["conferenceseries"] = self.l.fit_transform(self.d.data["conferenceseries"])
                self.classes = np.array(self.l.classes_)
            else:
                # update labels with IDs given by LabelEncoder
                for i, c in enumerate(classes):
                    self.d.data.loc[self.d.data["conferenceseries"]==c,"conferenceseries"] = i
                # set label to -1 if not in training data
                self.d.data.loc[pd.to_numeric(self.d.data["conferenceseries"], errors="coerce").isnull(),"conferenceseries"] = -1
                self.classes = classes
                
            data_labels = np.array(self.d.data["conferenceseries"],dtype="int64")
            
            print("Saving classes.")
            file = os.path.join(self.filepath,"classes.pkl")
            with open(file,"wb") as f:
                pickle.dump(self.classes, f)
            
            print("Loading word embeddings and parser.")
            timer.tic()
            # load GloVe model
            self.glove = EmbeddingsParser()
            self.glove.load_model(glove_model)
            timer.toc()
            
            print("Preprocessing abstracts.")
            timer.tic()
            inputs = list(self.d.data["chapter_abstract"].str.lower())
            timer.toc()

            chunks_max = math.ceil(len(inputs)/chunk_size)
            print("Transforming and saving abstracts: {} chunks.".format(chunks_max))
            timer.tic()
            timer.set_counter(chunks_max,max=10)
            for chunk in np.arange(chunks_max):
                vectors = self.glove.transform_vectors(inputs[chunk*chunk_size:(chunk+1)*chunk_size])
                file = os.path.join(self.filepath,"data."+str(chunk))
                with open(file,"wb") as f:
                    # save labels with abstracts
                    pickle.dump([np.array(vectors),data_labels[chunk*chunk_size:(chunk+1)*chunk_size]], f)
                timer.count()
            print("... total time:")
            timer.toc()
            
            self.chunks_max = chunks_max
            self.chunk_size = chunk_size
            self.size = len(self.d.data)
            file = os.path.join(self.filepath,"meta.pkl")
            with open(file,"wb") as f:
                pickle.dump([self.chunks_max, self.chunk_size, self.size], f)
                
            del self.d
            del self.glove
            if hasattr(self,"l"):
                del self.l
        
    #def get(self,i):
    #    label = torch.cuda.LongTensor(self.data_labels[i]).view(1)
    #    x = torch.cuda.FloatTensor(self.data_inputs[i]).unsqueeze(0).unsqueeze(0)
    #
    #    return x, label
    
    def batchify(self,size,num_chunks,shuffle=True):
        """
        Batchifies the dataset.
        
        Args:
            size (int): number of rows in a batch.
            num_chunks (int): number of chunks to preload into memory.
            shuffle (bool): retrieve randomized batches
        """
        self.timer_all = 0 #delme
        
        self.batch_size = size
        # count total number of batches processed
        self.batch_total = 0
        
        self.chunk_current = 0
        self.chunks = np.arange(self.chunks_max)
        self.chunks_step = num_chunks
        if shuffle:
            np.random.shuffle(self.chunks)
        self.shuffle = shuffle
        
        self._next_chunks()
        
    def _next_chunks(self):
        if self.chunk_current < self.chunks_max:
            if VERBOSE_EPOCHS:
                print("Loading next chunks.")
                timer.tic()
            
            # don't reload the data if there is only one chunk
            if not (hasattr(self,"data_inputs") and self.chunks_max==1):
                self._load_chunks(self.chunks[self.chunk_current:self.chunk_current+self.chunks_step])
                
            # initialize batching
            self.batch_current = 0
            self.batch_max = len(self.data_inputs)/self.batch_size
            self.batches = np.arange(len(self.data_inputs))
            if self.shuffle:
                np.random.shuffle(self.batches)
            
            if VERBOSE_EPOCHS:
                timer.toc()
            self.chunk_current += self.chunks_step
            return True
        
        return False
        
    def _load_chunks(self,chunks):
        self._clear_memory()
        
        for c in chunks:
            file = os.path.join(self.filepath,"data."+str(c))
            with open(file,"rb") as f:
                inputs, labels = pickle.load(f)
                try:
                    self.data_inputs = np.concatenate((self.data_inputs,inputs))
                except AttributeError:
                    self.data_inputs = inputs
                try:
                    self.data_labels = np.concatenate((self.data_labels,labels))
                except AttributeError:
                    self.data_labels = labels
            
    def _clear_memory(self):
        if hasattr(self,"data_inputs"):
            del self.data_inputs
        if hasattr(self,"data_labels"):
            del self.data_labels
            
        #self.data_inputs = np.empty((self.chunks_step*self.chunk_size,),dtype="object")
        #self.data_labels = np.empty((self.chunks_step*self.chunk_size,),dtype="int64")
        
    def next_batch(self):
        if not self.batch_current < self.batch_max:
            self._next_chunks()
            
        #print("Getting batch {}/{}".format(self.batch_current,self.batch_max))
        #timer.tic()
        
        # get (shuffled) indices
        i = self.batch_current * self.batch_size
        indices = self.batches[i:(i+self.batch_size)]

        # get labels
        labels = self.data_labels[indices]
        labels = torch.cuda.LongTensor(labels).view(len(labels))#,1)
        
        # get abstracts
        #inputs = self.data_inputs[indices]
        inputs = list(self.data_inputs[indices])
        
        # pad inputs to max length
        max_len = max(len(l) for l in inputs)
        #timer.tic() #delme
        for i, inp in enumerate(inputs):
            inputs[i] = np.concatenate((inp,np.zeros(max_len-inp.size)))
        #self.timer_all += timer.toc() #delme
        
        #inputs = torch.cuda.FloatTensor(list(inputs)).unsqueeze(1)
        inputs = torch.cuda.FloatTensor(inputs).unsqueeze(1)
        
        self.batch_current += 1
        self.batch_total += 1
        
        #timer.toc()
        return inputs, labels
    
    def has_next_batch(self):
        return (self.batch_current < self.batch_max) or (self.chunk_current < self.chunks_max)
        
    def num_classes(self):
        return len(self.classes)



class Net(nn.Module):
    
    path_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "data",
            "processed",
            "cnn",
            "model-save-state"
    )
    
    def __init__(self,embedding_size,classes,filters=100):
        super(Net, self).__init__()
        
        self.EMB = embedding_size
        
        # 1 channel in, <filters> channels out
        self.conv1 = nn.Conv1d(1,filters,
                               5*self.EMB,
                               stride=1*self.EMB
        )
        self.conv2 = nn.Conv1d(1,filters,
                               4*self.EMB,
                               stride=1*self.EMB
        )
        self.conv3 = nn.Conv1d(1,filters,
                               3*self.EMB,
                               stride=1*self.EMB
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
    
    def save_state(self,epoch,losses,optimizer):
        self.model_state = {
                    "epoch":epoch,
                    "losses":losses,
                    "model":self.state_dict(),
                    "optimizer":optimizer.state_dict()
        }
        torch.save(self.model_state, self.path_persistent)
        
    def load_state(self,optimizer):
        self.model_state = torch.load(self.path_persistent)
        
        self.load_state_dict(self.model_state["model"])
        optimizer.load_state_dict(self.model_state["optimizer"])
        
        return self.model_state["epoch"], self.model_state["losses"]
    
    def plot_losses(self):
        ymax = max((max(self.model_state["losses"][0]),max(self.model_state["losses"][1])))
        plt.plot(self.model_state["losses"][0])
        plt.plot(self.model_state["losses"][1])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend(["train","eval"])
        plt.ylim(ymin=0,ymax=ymax+0.5)
        plt.show()
        
        

print(">>> loading data")

timer = Timer()
d_train = BatchifiedData(
        data_which=DATA_TRAIN,
        glove_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE
)
gc.collect()
d_test = BatchifiedData(
        training=False,
        classes=d_train.classes,
        data_which=DATA_TEST,
        glove_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE
)
gc.collect()

print(">>> creating net")

# create Net
net = Net(
        embedding_size=EMBEDDING_SIZE,
        classes=d_train.num_classes(),
        filters=NUM_FILTERS_CONV
)
net.cuda()

# create optimizer
#optimizer = optim.SGD(net.parameters(), lr=1)
#optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)
optimizer = optim.Adadelta(
        net.parameters()
        ,weight_decay=WEIGHT_DECAY
)

losses_train = []
losses_test = []

#for param in net.conv1.parameters():
#    print(param)

print(">>> starting training")

# batch data loading
for epoch in range(EPOCHS):
    print("============ EPOCH {} ============".format(epoch))
    
    ### TRAINING
    net.train()
    
    running_loss = 0
    divisor = 0
    timer.tic()
    if VERBOSE_EPOCHS:
        timer.set_counter(d_train.size,max=VERBOSE_EPOCHS_STEPS)
    
    if VERBOSE_EPOCHS:
        print("Batchify.")
    d_train.batchify(BATCH_SIZE,CHUNKS_IN_MEMORY,shuffle=SHUFFLE_TRAIN)
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
    timer.toc()
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
        timer.set_counter(d_test.size,max=VERBOSE_EPOCHS_STEPS)
    
    d_test.batchify(BATCH_SIZE,CHUNKS_IN_MEMORY,shuffle=False)
    while d_test.has_next_batch():
        inputs, labels = d_test.next_batch()
        
        outputs = net(inputs)
        loss = net.loss(outputs,labels)
        
        running_loss += loss.item()
        if VERBOSE_EPOCHS:
            timer.count(len(inputs))
        
    running_loss = running_loss/math.ceil(d_test.batch_total)
    print("Eval ==> Epoch: {}, Loss: {}".format(epoch,running_loss))
    timer.toc()
    losses_test.append(running_loss)

#save model state
net.save_state(epoch,[losses_train,losses_test],optimizer)

# draw losses
net.plot_losses()

#for param in net.conv1.parameters():
#    print(param)
#    print(param.grad.data.sum())