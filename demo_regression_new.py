import sys
import time
import pickle
import numpy as np
sys.path.append('../nn')

import network_new
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint
import torch
from AnimationPlot import animation_plot
rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_valid
X = np.swapaxes(X, 1, 2)
preprocess = np.load('preprocess_core.npz')
preprocess_footstepper = np.load('preprocess_footstepper.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1

def decoder(autoencoder, Upsilon_T):
    x = autoencoder.unpool1d(Upsilon_T)
    x = autoencoder.dropout2(x)
    x = autoencoder.conv2(x)
    return x

def create_network(window, input):
    network_feedforward = network_new.Feedforward()
    network_feedforward.load_state_dict(torch.load("model_dir/model_feedforward250.pth.tar",map_location=torch.device('cpu')))
    network_autoencoder = network_new.AutoEncoder()
    network_autoencoder.load_state_dict(torch.load("model_dir/model_100.pth.tar",map_location=torch.device('cpu')))
    for p in network_feedforward.parameters():
        p.requires_grad = False
    for p in network_autoencoder.parameters():
        p.requires_grad = False
    return network_feedforward, network_autoencoder

#print("X.shape: ", X.shape)

indices = [(30, 15*480), (60, 15*480), (90, 15*480)]

for index, length in indices:
    Torig = np.load('../data/curves.npz')['C'][:,:,index:index+length]
    Torig = (Torig - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]
    network_footstepper = network_new.LocomotionNetwork()
    network_footstepper.load_state_dict(torch.load("model_dir/model_locomotion200.pth.tar",map_location=torch.device('cpu')))
    for p in network_footstepper.parameters():
        p.requires_grad = False
    start = time.clock()
    W = network_footstepper(torch.from_numpy(Torig[:,:3]).to(dtype=torch.float))
    W = W.cpu().numpy()
    W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']
    # alpha - user parameter scaling the frequency of stepping.
    #         Higher causes more stepping so that 1.25 adds a 
    #         quarter more steps. 1 is the default (output of
    #         footstep generator)
    #
    # beta - Factor controlling step duration. Increasing reduces 
    #        the step duration. Small increases such as 0.1 or 0.2 can
    #        cause the character to run or jog at low speeds. Small 
    #        decreases such as -0.1 or -0.2 can cause the character 
    #        to walk at high speeds. Too high values (such as 0.5) 
    #        may cause the character to skip steps completely which 
    #        can look bad. Default is 0.
    #
    #alpha, beta = 1.25, 0.1
    alpha, beta = 1.0, 0.0
    
    # controls minimum/maximum duration of steps
    minstep, maxstep = 0.9, -0.5
    
    off_lh, off_lt, off_rh, off_rt = 0.0, -0.1, np.pi+0.0, np.pi-0.1

    Torig = (np.concatenate([Torig,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lh)>np.clip(np.cos(W[:,1:2])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lt)>np.clip(np.cos(W[:,2:3])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rh)>np.clip(np.cos(W[:,3:4])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rt)>np.clip(np.cos(W[:,4:5])+beta, maxstep, minstep))*2-1], axis=1))
 
    # print('Footsteps: %0.4f' % (time.clock() - start))
    # print("Torig.shape: ", Torig.shape) # (1, 7, 7200)
    #############
    network_feedforward, network_autoencoder = create_network(Torig.shape[2], Torig.shape[1])
    
    start = time.clock()
    Xrecn = decoder(network_autoencoder, network_feedforward(torch.from_numpy(Torig).to( dtype=torch.float)))
    Xrecn = Xrecn.cpu().numpy()
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xtraj = ((Torig * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]).copy()
    #print('Synthesis: %0.4f' % (time.clock() - start))
    
    Xnonc = Xrecn.copy() # (1, 73, 7200)
    print("Xnonc.shape: ", Xnonc.shape)
    Xrecn = constrain(Xrecn, network_autoencoder, preprocess, multiconstraint(
        foot_sliding(Xtraj[:,-4:]),
        trajectory(Xtraj[:,:3]),
        joint_lengths()), alpha=0.01, iterations=250)
    Xrecn[:,-7:] = Xtraj
    
    animation_plot([Xrecn], interval=15.15)
    #animation_plot([Xrecn], interval=15.15)

