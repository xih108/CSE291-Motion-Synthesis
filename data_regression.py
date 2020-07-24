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
rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_train
X = np.swapaxes(X, 1, 2)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

Y = X[:,-7:]

I = np.arange(len(X))
rng.shuffle(I)
X, Y = X[I], Y[I]

print(X.shape, Y.shape)
class DATA_REGRESSION(Dataset):
    def __init__(self, mode='train'):
        self.label = torch.from_numpy(X.astype(np.float32))
        self.data = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        inputs = {
            'X': self.label[idx],
            'Y': self.data[idx]
        }
        return inputs