import gc

import torch
import torch.optim as optim

#sys.path.insert(0, os.path.join(os.getcwd(),"..","data"))

from TimerCounter import Timer

#### TRAINING PARAMETERS ####

USE_CUDA = False

# set a name which is used as directory for the save states in "data/processed/nn/<name>"
NET_NAME = "RNN-100"

EPOCHS = 10
DATA_TRAIN = "small"
SHUFFLE_TRAIN = True
DATA_TEST = "small"
NUM_HIDDEN = 100
WEIGHT_DECAY = 0.0005

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

from BatchifiedCharactersData import BatchifiedCharactersData
from RNNet import RNNet

print(">>> loading data")

timer = Timer()
d_train = BatchifiedCharactersData(
        use_cuda=USE_CUDA,
        data_which=DATA_TRAIN
)
gc.collect()
print("pre-d_test")
d_test = BatchifiedCharactersData(
        use_cuda=USE_CUDA,
        training=False,
        classes=d_train.classes,
        data_which=DATA_TEST
)
print("post-d_test")
gc.collect()

print(">>> creating net")

# create Net
net = RNNet(
        NET_NAME,
        letters_size=d_train.num_letters(),
        num_classes=d_train.num_classes(),
        hidden_size=NUM_HIDDEN
)
if USE_CUDA:
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
    timer.tic()
    if VERBOSE_EPOCHS:
        timer.set_counter(d_train.data_size,max=VERBOSE_EPOCHS_STEPS)
    
    if VERBOSE_EPOCHS:
        print("Batchify.")
    d_train.shuffle()
    while d_train.has_next():
        #print("next_item: {}".format(d_train.current_item))
        input, label = d_train.next_item()
        print(input.size())
        print(label)
        if input.size()[0]>1000:
            continue
        
        optimizer.zero_grad()
        output = net(input)
        print(output.size())
        loss = net.loss(output.unsqueeze(0),label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if VERBOSE_EPOCHS:
            timer.count()
        
    running_loss = running_loss/d_train.data_size
    print("Train ==> Epoch: {}, Loss: {}".format(epoch,running_loss))
    net.training_time += timer.toc()
    losses_train.append(running_loss)
    
    ### EVALUATION
    net.eval()
    
    running_loss = 0
    timer.tic()
    if VERBOSE_EPOCHS:
        timer.set_counter(d_test.data_size,max=VERBOSE_EPOCHS_STEPS)
    
    d_test.shuffle()
    while d_test.has_next():
        input, label = d_test.next_item()
        
        with torch.no_grad():
            output = net(input)
            loss = net.loss(output.unsqueeze(0),label)
        
        running_loss += loss.item()
        if VERBOSE_EPOCHS:
            timer.count()
        
    running_loss = running_loss/d_test.data_size
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