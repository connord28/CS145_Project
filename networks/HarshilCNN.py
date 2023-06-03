# libraries we need
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class HarshilCNN(nn.Module):

    def __init__(self, channels, pic_size, num_classes):
        super(HarshilCNN, self).__init__()
        # input dim: channels x pic_size x pic_size = c x p X p
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.6))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=0.4))
        
        self.fc1 = nn.Linear(107648, 625, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.4))

        self.fc2 = torch.nn.Linear(625, num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)   # Flatten them for FC
        x = self.fc1(x)
        x = self.fc2(x)
        return x
