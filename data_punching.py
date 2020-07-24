import os
import sys
import numpy as np
import scipy.io as io
import json
import torch
import scipy.misc
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

data = np.load('../data/processed/data_hdm05.npz')
data_punching = np.hstack([np.arange(259, 303), np.arange(930, 978), np.arange(1650,1703), np.arange(2243,2290), np.arange(2851,2895)])

punching_train = data_punching[:len(data_punching)//2]
punching_valid = data_punching[len(data_punching)//2:]

X = data['clips'][punching_train]
X = np.swapaxes(X, 1, 2)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

hands = np.array([60,61,62,48,49,50])
Y1 = X[:,hands]
Y2 = np.zeros((Y1.shape[0], 1, Y1.shape[2])) #concatenate Y to make Y.shape[1] = 7
Y = np.concatenate((Y1, Y2), axis=1)

batchsize = 1
window = X.shape[2]

print(X.shape, Y.shape)
class DATA_PUNCHING(Dataset):
    def __init__(self, mode='train'):
        self.label = torch.from_numpy(X.astype(np.float32))
        self.data = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = {
            'X': self.label[idx],
            'Y': self.data[idx]
        }
        return inputs