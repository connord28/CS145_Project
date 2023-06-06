import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class StartingCNN5(nn.Module):

    def __init__(self, channels, pic_size, num_classes):
        super(StartingCNN5, self).__init__()
        # Modifying from StartingCNN2/3 with inspiration from https://arxiv.org/pdf/1409.1556.pdf using smaller 3x3 kernels and many layers
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=3, stride=1, padding=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=2)
        self.c3 = nn.Conv2d(12, 36, 3, 1, 2)
        self.c4 = nn.Conv2d(36, 36, 3, 1, 2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat = nn.Flatten()
        self.d1 = nn.Linear(36*14*14, 512)
        self.d2 = nn.Linear(512, 512)
        self.d3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = self.pool(F.relu(self.c3(x)))
        x = self.pool(F.relu(self.c4(x)))

        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)
        return x