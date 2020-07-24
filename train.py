import os
import sys
import torch
import network_new
import data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    

save_dir = 'model_dir'
random_seed = 0
gpu_id = 0
epochs = 200

if __name__=='__main__':
    torch.cuda.set_device(gpu_id)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    print('=====>Prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(mode='train'),
                                               batch_size=1,
                                               shuffle=True)
    print('=====>Prepare model ...')
    model = network_new.AutoEncoder()
    model.cuda()
    #optimizer and log writter
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    writer = SummaryWriter(os.path.join(save_dir, 'train_info'))
    print('Start training ...')
    print('=====>Start training ...')
    iters = 0
    model.train()
    for epoch in range(1, epochs+1):     
        train_loss = 0

        for idx, motion in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1
            motion = motion.cuda()
            output = model(motion)
            # print("output shape: ", output.shape)
            # compute loss, backpropagation, update parameters 
            criterion = nn.MSELoss()
            alpha = 0.0001
            L1loss = 0
            for p in model.parameters():
                L1loss = L1loss + p.abs().sum()
            mseloss = criterion(output, motion) # compute loss
            loss = mseloss + alpha*L1loss
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters
            train_loss += loss/len(train_loader)
            # logger
            if iters%1000 == 0:    
                writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                #train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
                #sys.stdout.write('\r')
                #sys.stdout.write(train_info)
                #sys.stdout.flush()
        print("epoch ", epoch, ": loss = ", train_loss)
        print('=====>Save model ...')
        save_model(model, os.path.join(save_dir, 'model_{}.pth.tar'.format(epoch)))
