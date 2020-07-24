import sys
import numpy as np
import scipy.io as io
sys.path.append('../nn')
import torch
import network_new
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

X = np.load('../data/processed/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1
window = X.shape[2]

network_autoencoder = network_new.AutoEncoder()
network_autoencoder.load_state_dict(torch.load("model_dir/model_100.pth.tar",map_location=torch.device('cpu')))
for p in network_autoencoder.parameters():
    p.requires_grad = False

# 2021
# 13283
for _ in range(10):
    index = rng.randint(X.shape[0])
    print(index)
    Xorgi = X[index:index+1]
    Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5)
    Xrecn = network_autoencoder(torch.from_numpy(Xnois).to(dtype=torch.float))
    Xrecn = Xrecn.cpu().numpy()
    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xnonc = np.copy(Xrecn) # no constraint
    Xrecn = constrain(Xrecn, network_autoencoder, preprocess, multiconstraint(
        foot_sliding(Xorgi[:,-4:].copy()),
        joint_lengths(),
        trajectory(Xorgi[:,-7:-4])), alpha=0.01, iterations=250)

    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]
    Xnonc[:,-7:-4] = Xorgi[:,-7:-4]    
    # animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    if(np.array_equal(Xnonc, Xrecn)):
        print("same with constraint")