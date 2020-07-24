import sys
import numpy as np
import scipy.io as io
import os
import torch
from network_new import *
from data_punching import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

batchsize = 1
window = X.shape[2]

network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network_second.load(np.load('network_core.npz'))

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1])
network = Network(network_first, network_second[1], params=network_first.params)

E = theano.shared(X, borrow=True)
F = theano.shared(Y, borrow=True)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network, F, E, filename='network_regression_punch.npz')



def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  


def decoder(autoencoder, Upsilon_T):
    x = autoencoder.unpool1d(Upsilon_T)
    x = autoencoder.dropout2(x)
    x = autoencoder.conv2(x)
    return x


save_dir = 'model_dir'


if __name__=='__main__':
    print('=====>Prepare dataloader ...')
    data = DATA_PUNCHING(mode='train')
    X = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)
    train_loader = torch.utils.data.DataLoader(data[:,-7:],batch_size=1,shuffle=True)

    print('=====>Prepare model ...')
    autoencoder = AutoEncoder().cuda()
    autoencoder.load_state_dict(torch.load("model_dir/model_100.pth.tar",map_location=torch.device('cpu')))
    # autoencoder.eval()
    # freeze weights for autoencoder
    for p in autoencoder.parameters():
        p.requires_grad = False

    feedforward = Feedforward().cuda()
    #Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(feedforward.parameters(),lr = 0.001)
    writer = SummaryWriter(os.path.join(save_dir, 'train_info'))

    alpha = 0.0001
    epoch_num = 100
    iters = 0

    for epoch in range(epoch_num):
        train_loss = 0
        
        for idx, (Upsilon_T, data_X) in enumerate(zip(train_loader, X)):
            iters += 1
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch+1, idx+1, len(train_loader))
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            # Upsilon_T, encoded_X = Upsilon_T.to(computing_device), encoded_X.to(computing_device)
            data_X = data_X.cuda()
            Upsilon_T = Upsilon_T.cuda()
            outputs_T = feedforward(Upsilon_T)
            criterion = nn.MSELoss()
            decoded_T = decoder(autoencoder, outputs_T)

            # loss = criterion(data_X, decoded_T) + alpha*np.sum(np.abs(omega))
            loss = criterion(data_X, decoded_T)
            L1loss = 0
            for p in feedforward.parameters():
                L1loss = L1loss + p.abs().sum()
            loss = loss + alpha*L1loss
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()    
            # Add this iteration's loss to the total_loss
            if iters%1000 == 0:    
                    writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                    train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
                    sys.stdout.write('\r')
                    sys.stdout.write(train_info)
                    sys.stdout.flush()
        print('=====>Save model ...')
        save_model(feedforward, os.path.join(save_dir, 'model_feedforward{}.pth.tar'.format(epoch)))