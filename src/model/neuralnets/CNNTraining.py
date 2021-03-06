import gc

import torch.optim as optim

import math
#sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))

from TimerCounter import Timer

#### CUDA

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

#### TRAINING PARAMETERS ####

# set a name which is used as directory for the save states in "data/processed/nn/<name>"
NET_NAME = "CNN-CPU-test"

BATCH_SIZE = 500                    # 300d: 150
CHUNKS_IN_MEMORY = 5                # 300d: 3
CHUNK_SIZE = 10000                  # 300d: 5000
EPOCHS = 10
DATA_TRAIN = "medium"
SHUFFLE_TRAIN = False
DATA_TEST = "small"
EMBEDDING_MODEL = "6d50"
EMBEDDING_SIZE = 50
NUM_FILTERS_CONV = 50
WEIGHT_DECAY = 0.0005

VERBOSE_EPOCHS = False
VERBOSE_EPOCHS_STEPS = 10

# keep states of each epoch of the net
KEEP_STATES = True

# load the last saved state and continue training
CONTINUE_TRAINING = True

#############################

from BatchifiedEmbeddingsData import BatchifiedEmbeddingsData
from CNNet import CNNet

print(">>> loading data")

timer = Timer()
d_train = BatchifiedEmbeddingsData(
        data_which=DATA_TRAIN,
        glove_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE
)
gc.collect()
d_test = BatchifiedEmbeddingsData(
        training=False,
        classes=d_train.classes,
        data_which=DATA_TEST,
        glove_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE
)
gc.collect()

print(">>> creating net")

# create Net
net = CNNet(
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
epoch = 0

if CONTINUE_TRAINING and net.save_state_exists():
    epoch, [losses_train, losses_test] = net.load_state(optimizer)
    epoch += 1
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