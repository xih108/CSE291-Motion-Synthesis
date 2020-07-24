import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.conv1 = nn.Conv1d(73, 256, kernel_size=25, stride=1, padding=12, bias=True) # 73x240 -> 256x240,   kernel size=25
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True) # 256x240 -> 256x120, which is latent space
        # decoder
        self.unpool1d = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1, bias=True) # 256x120 -> 256x240, what kernel size
        # self.unpool1d = nn.MaxUnpool1d(kernel_size=2, stride=2) # 256x240
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(256, 73, kernel_size=25, stride=1, padding=12, bias=True) # 256x240 -> 73,240   kernel size=25?
    
    def forward(self, motion):
        x = self.conv1(motion)
        x = self.relu1(x)
        x = self.dropout1(x)
        x, indices = self.maxpool1(x)
        # x = self.unpool1d(x, indices)
        x = self.unpool1d(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        return x


class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(7, 64 , kernel_size = 45, padding = 22, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128,  kernel_size = 25, padding = 12,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256 , kernel_size = 15, padding = 7, bias=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.ReLU(inplace=True),  
        )
        
    def forward(self, input):
        return self.main(input)


class LocomotionNetwork(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(LocomotionNetwork, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=65, padding=32, bias=True)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv2 = nn.Conv1d(64, 5, kernel_size=45, padding=22, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        
        return x