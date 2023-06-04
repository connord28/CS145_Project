# libraries we need
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class StartingCNN3(nn.Module):

    def __init__(self, channels, pic_size, num_classes):
        super(StartingCNN3, self).__init__()

        self.c1 = nn.Conv2d(in_channels=channels, out_channels=12, kernel_size=6, stride=1, padding=2)
        self.c2 = nn.Conv2d(12, 36, 6, 1, 2)

        self.pool = nn.MaxPool2d(kernel_size=6, stride=2)


        self.flat = nn.Flatten()
        self.d1 = nn.Linear(36*52*52, 120)
        self.d2 = nn.Linear(120, 84)
        self.d3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))

        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x) # no need for softmax
        return x