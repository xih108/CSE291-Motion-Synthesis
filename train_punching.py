import os
import sys
import torch
from network_new import *
from data_punching import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

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
    train_loader = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)

    print('=====>Prepare model ...')
    autoencoder = AutoEncoder().cuda()
    autoencoder.load_state_dict(torch.load("model_dir/model_100.pth.tar",map_location=torch.device('cpu')))
    # autoencoder.eval()
    # freeze weights for autoencoder
    for p in autoencoder.parameters():
        p.requires_grad = False

    feedforward = Feedforward().cuda()
    #Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
    optimizer = optim.Adam(feedforward.parameters(),lr = 1e-4)
    writer = SummaryWriter(os.path.join(save_dir, 'train_info'))

    #alpha = 0.0001
    alpha = 0
    epochs = 250
    iters = 0

    for epoch in range(1, epochs+1):
        train_loss = 0
        
        for idx, batch in enumerate(train_loader):
            #X, Y = batch['X'].cuda().float(), batch['Y'].cuda().float()
            X = batch['X'].cuda()
            Y = batch['Y'].cuda()
            iters += 1
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch+1, idx+1, len(train_loader))
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            # Upsilon_T, encoded_X = Upsilon_T.to(computing_device), encoded_X.to(computing_device)
            outputs_T = feedforward(Y)
            criterion = nn.MSELoss()
            decoded_T = decoder(autoencoder, outputs_T)
            loss = criterion(X, decoded_T)
            L1loss = 0
            for p in feedforward.parameters():
                L1loss = L1loss + p.abs().sum()
            loss = loss + alpha*L1loss
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()
            # Update the weights
            optimizer.step()
            train_loss += loss/len(train_loader)
            # Add this iteration's loss to the total_loss
            if iters%1000 == 0:    
                    writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                    #train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
                    #sys.stdout.write('\r')
                    #sys.stdout.write(train_info)
                    #sys.stdout.flush()
        print("epoch ", epoch, ": loss = ", train_loss)
        print('=====>Save model ...')
        save_model(feedforward, os.path.join(save_dir, 'punching_feedforward{}.pth.tar'.format(epoch)))

