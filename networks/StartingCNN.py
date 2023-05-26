# libraries we need
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# construct your CNN model
# we've provided the basic structure of the class you need to implement
# replace all the 'None' statements and add in the architecture of your CNN
# 20 minutes
class StartingCNN(nn.Module):

    def __init__(self):
        super(StartingCNN, self).__init__()
        # input dim: 3x32x32
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        # output dim: 6x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output dim: 6x16x16
        self.c2 = nn.Conv2d(6, 16, 5, 1, 2)
        # output dim: 16x16x16
        self.pool2 = nn.MaxPool2d(2, 2)
        # output dim: 16x8x8
        self.flat = nn.Flatten()
        self.d1 = nn.Linear(16*8*8, 120)
        self.d2 = nn.Linear(120, 84)
        self.d3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.c1(x)))
        x = self.pool2(F.relu(self.c2(x)))
        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x) # no need for softmax
        return x