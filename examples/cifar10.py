import torch
import torch.nn as nn
from tfutils.fp16 import convertToFP16
from tfutils.initializers import *
from tfutils.metrics import Accuracy
from tfutils.models import load_model,save_model
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

CUDA = False

if torch.cuda.is_available():
    CUDA = True

BATCH_SIZE = 256
NUM_EPOCHS = 100

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):

        return self.net(x)


class CifarNet(nn.Module):
    def __init__(self,num_classes=10):
        super(CifarNet,self).__init__()

        self.net = nn.Sequential(

            ConvBlock(3,32),
            ConvBlock(32,32),
            ConvBlock(32,32),

            nn.MaxPool2d(3,2),

            ConvBlock(32,64),
            ConvBlock(64,64),
            ConvBlock(64,64),

            nn.MaxPool2d(3,2),

            ConvBlock(64,128),
            ConvBlock(128,128),
            ConvBlock(128,128),

            nn.AdaptiveAvgPool2d((1,1)),
        )

        self.classifier = nn.Linear(128,num_classes)

    def forward(self,x):

        x = self.net(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

model = CifarNet()

if CUDA:
    model = model.cuda()

#initialize the weights
kaiming_normal_init(model,types=[nn.Conv2d])
xavier_normal_init(model,types=[nn.Linear])

#initialize the bias
zeros_init(model,category="bias")

#setup optimizer

optimizer = Adam(model.parameters())

#convert to mixed precision mode

model,optimizer = convertToFP16(model,optimizer)

#create your lr scheduler
lr_scheduler = StepLR(optimizer,step_size=30,gamma=0.1)

#create data transforms
train_transforms = transforms.Compose([

    transforms.RandomCrop(size=(32,32),padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))

])

test_transforms = transforms.Compose([

    transforms.CenterCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))

])

#load the dataset
train_dataset = CIFAR10(root="./cifar10",train=True,transform=train_transforms,download=True)
test_dataset = CIFAR10(root="./cifar10",train=False,transform=test_transforms,download=True)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE)


#evaluation function
def evaluation_loop():

    model.eval()

    test_acc = Accuracy()

    for i,(x,y) in enumerate(test_loader):
        
        x = x.cuda()
        y = y.cuda()

        predictions = model(x)

        test_acc.update(predictions,y)
    return test_acc.getValue()

#training function
def train_loop():

    loss_fn = nn.CrossEntropyLoss()

    train_acc = Accuracy()

    best_acc = 0

    for e in range(NUM_EPOCHS):

        model.train()

        train_acc.reset()

        total_loss = 0

        for i,(x,y) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()

            if CUDA:
                x = x.cuda()
                y = y.cuda()

            predictions = model(x)
            loss = loss_fn(predictions,y)

            #replace loss.backward() with optimizer.backward(loss)
            optimizer.backward(loss)
            
            optimizer.step()

            total_loss = total_loss + (loss.item() * x.size(0))

            train_acc.update(predictions,y)

        total_loss = total_loss/len(train_loader.dataset)

        lr_scheduler.step()

        test_acc = evaluation_loop(model)

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model,"model_{}.pth".format(e))
    
        print("Epoch: {} Train Acc: {} Test Acc: {} Best Test Acc: {} Loss: {}".format(e,train_acc.getValue(),test_acc,best_acc,total_loss))


train_loop()









