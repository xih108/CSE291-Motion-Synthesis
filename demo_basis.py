import sys
import numpy as np
import scipy.io as io
from network_new import *

from AnimationPlot import animation_plot


sys.path.append('../nn')

rng = np.random.RandomState(23455)

preprocess = np.load('preprocess_core.npz')


batchsize = 1
window = 240



network = AutoEncoder().cuda()
network.load_state_dict(torch.load("./model_100.pth.tar",map_location=torch.device('cpu')))
for p in network.parameters():
        p.requires_grad = False

def decoder(autoencoder, Upsilon_T):
    x = autoencoder.unpool1d(Upsilon_T)
    x = autoencoder.dropout2(x)
    x = autoencoder.conv2(x)
    return x


for i in range(10):
    Xbasis0 = np.zeros((1, 256, window//2))
    Xbasis1 = np.zeros((1, 256, window//2))
    Xbasis2 = np.zeros((1, 256, window//2))
    
    Xbasis0[:,i*3+0] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis1[:,i*3+1] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis2[:,i*3+2] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    
    Xbasis0 = decoder(network, torch.from_numpy(Xbasis0).cuda().to( dtype=torch.float))
    Xbasis1 = decoder(network, torch.from_numpy(Xbasis1).cuda().to( dtype=torch.float))
    Xbasis2 = decoder(network, torch.from_numpy(Xbasis2).cuda().to( dtype=torch.float))    

    Xbasis0 = Xbasis0.cpu().numpy()
    Xbasis1 = Xbasis1.cpu().numpy()
    Xbasis2 = Xbasis2.cpu().numpy()

    Xbasis0 = (Xbasis0 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis1 = (Xbasis1 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis2 = (Xbasis2 * preprocess['Xstd']) + preprocess['Xmean']
        
    animation_plot([Xbasis0, Xbasis1, Xbasis2], interval=15.15)
        