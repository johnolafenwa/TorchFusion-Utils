import torch
import torch.nn as nn
from tfutils.fp16 import convertToFP16
from tfutils.initializers import *
from tfutils.metrics import Accuracy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


BATCH_SIZE = 32
NUM_EPOCHS = 100

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

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
        x = x.view(-1)
        return self.classifier(x)

t1 = torch.FloatTensor(10,3,32,32)

m = CifarNet()

out = m(t1)
print(out.size())
