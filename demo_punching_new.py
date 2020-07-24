import sys
import time
import numpy as np
import scipy.io as io
sys.path.append('../nn')
import network_new
# from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint
import torch
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)
data = np.load('../data/processed/data_hdm05.npz')
data_punching = np.hstack([np.arange(259, 303), np.arange(930, 978), np.arange(1650,1703), np.arange(2243,2290), np.arange(2851,2895)])
rng.shuffle(data_punching)

punching_train = data_punching[:len(data_punching)//2]
punching_valid = data_punching[len(data_punching)//2:]

#X = data['clips'][punching_valid]
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

def decoder(autoencoder, Upsilon_T):
    x = autoencoder.unpool1d(Upsilon_T)
    x = autoencoder.dropout2(x)
    x = autoencoder.conv2(x)
    return x

def create_network(window, input):
    network_feedforward = network_new.Feedforward()
    network_feedforward.load_state_dict(torch.load("model_dir/punching_feedforward250.pth.tar",map_location=torch.device('cpu')))
    network_autoencoder = network_new.AutoEncoder()
    network_autoencoder.load_state_dict(torch.load("model_dir/model_100.pth.tar",map_location=torch.device('cpu')))
    for p in network_feedforward.parameters():
        p.requires_grad = False
    for p in network_autoencoder.parameters():
        p.requires_grad = False
    return network_feedforward, network_autoencoder

for i in range(len(X)):
    Torig = Y[i:i+1]
    network_feedforward, network_autoencoder = create_network(Torig.shape[2], Torig.shape[1])
    start = time.clock()
    Xrecn = decoder(network_autoencoder, network_feedforward(torch.from_numpy(Torig).to( dtype=torch.float)))
    Xrecn = Xrecn.cpu().numpy()
    Xorig = np.array(X[i:i+1])
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    #Xrecn = constrain(Xrecn, network_second[0], network_second[1], preprocess, multiconstraint(
    #    foot_sliding(Xrecn[:,-4:].copy()),
    #    joint_lengths()), alpha=0.01, iterations=50)
    animation_plot([Xorig, Xrecn], interval=15.15)

