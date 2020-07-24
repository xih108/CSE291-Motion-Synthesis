import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class LocomotionDataset(Dataset):
    def __init__(self, data_path, preprocess_path, is_train=False):
        super(LocomotionDataset, self).__init__()
        data = np.load(data_path)['clips']
        rng = np.random.RandomState(23455)
        #I = np.arange(len(data))
        #rng.shuffle(I)
        #data = data[I[:len(data)//2]] #training data
        preprocess = np.load(preprocess_path)
        self.T, self.F, self.W = self.preprocess_data(data, preprocess)
        self.is_train = is_train

        # TODO: data augmentation

    def __len__(self):
        assert len(self.T) == len(self.F) == len(self.W)
        return len(self.T)

    def __getitem__(self, index):
        inputs = {
            'T': self.T[index],
            'W': self.W[index]
        }
        return inputs

    def preprocess_data(self, data, preprocess):
        X = data
        X = np.swapaxes(X, 1, 2).astype(np.float32)

        X = (X - preprocess['Xmean']) / preprocess['Xstd']

        T = X[:,-7:-4] # T: forward facing direction of the path, len=3
        F = X[:,-4:]   # F: matrix that represents the contact states of left heel, left toe, right heel, and right toe at each frame, len=4

        W = np.zeros((F.shape[0], 5, F.shape[2]))

        for i in range(len(F)):
            
            w = np.zeros(F[i].shape)
            
            for j in range(F[i].shape[0]):
                last = -1
                for k in range(1, F[i].shape[1]):
                    if last == -1 and F[i,j,k-1] < 0 and F[i,j,k-0] > 0: last = k; continue
                    if last == -1 and F[i,j,k-1] > 0 and F[i,j,k-0] < 0: last = k; continue
                    if F[i,j,k-1] > 0 and F[i,j,k-0] < 0:
                        if k-last+1 > 10 and k-last+1 < 60:
                            w[j,last:k+1] = np.pi/(k-last)
                        else:
                            w[j,last:k+1] = w[j,last-1]
                        last = k
                        continue
                    if F[i,j,k-1] < 0 and F[i,j,k-0] > 0:
                        if k-last+1 > 10 and k-last+1 < 60:
                            w[j,last:k+1] = np.pi/(k-last)
                        else:
                            w[j,last:k+1] = w[j,last-1]
                        last = k
                        continue
            
            c = np.zeros(F[i].shape)
            
            for k in range(0, F[i].shape[1]):
                window = slice(max(k-100,0),min(k+100,F[i].shape[1]))
                ratios = (
                    np.mean((F[i,:,window]>0).astype(np.float), axis=1) / 
                    np.mean((F[i,:,window]<0).astype(np.float), axis=1))
                ratios[ratios==np.inf] = 100
                c[:,k] = ((np.pi*ratios) / (1+ratios))
            
            w[w==0.0] = np.nan_to_num(w[w!=0.0].mean())
            
            W[i,0:1] = w.mean(axis=0)
            W[i,1:5] = c
            
            # import matplotlib.pyplot as plt
            # plt.plot(F[i,0])
            # plt.plot(np.sin(np.cumsum(W[i,0:1])))
            # plt.ylim([-1.1, 1.1])
            # plt.show()

        Wmean = W.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
        Wstd = W.std(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
        W = (W - Wmean) / Wstd

        print('T.shape:', T.shape)
        print('F.shape:', F.shape)
        print('W.shape:', W.shape)

        return T, F, W


