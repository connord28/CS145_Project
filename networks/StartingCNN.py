# libraries we need
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class StartingCNN(nn.Module):

    def __init__(self, channels, pic_size, num_classes):
        super(StartingCNN, self).__init__()
        # input dim: channels x pic_size x pic_size = c x p X p
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        # output dim: 2c x p x p
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output dim: 2c x p/2 x p/2
        self.c2 = nn.Conv2d(channels*2, 16, 5, 1, 2)
        # output dim: 16 x p/2 x p/2
        self.pool2 = nn.MaxPool2d(2, 2)
        # output dim: 16 x p/4 x p/4
        self.flat = nn.Flatten()
        self.d1 = nn.Linear(16*(pic_size//4)*(pic_size//4), 120)
        self.d2 = nn.Linear(120, 84)
        self.d3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.c1(x)))
        x = self.pool2(F.relu(self.c2(x)))
        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x) # no need for softmax
        return x
