import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_locomotion import LocomotionDataset
from tensorboardX import SummaryWriter
from network_new import LocomotionNetwork
import sys

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  

if __name__ == "__main__":
    batchsize = 1
    n_workers = 8
    max_epochs = 200
    log_path = "./locomotion_logs"

    random_seed = 1732
    gpu_id = 0
    # data_path = './data/data_edin_locomotion.npz'
    data_path = '/home/derek/Documents/motion_syn_official/data/processed/data_edin_locomotion.npz'
    preprocess_path = 'preprocess_core.npz'

    # set seed
    torch.cuda.set_device(gpu_id)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # data loader
    dataset = LocomotionDataset(data_path, preprocess_path)
    train_loader = DataLoader(dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True,
                              drop_last=True)

    # model
    net = LocomotionNetwork()
    net.cuda()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(os.path.join(log_path, 'train'))

    print('Begin training')
    for epoch in range(1, max_epochs+1):
        train_loss = 0
        for i, batch in enumerate(train_loader):
            T, W = batch['T'].cuda().float(), batch['W'].cuda().float()
            pred = net(T)
            loss = criterion(pred, W)

            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters
            train_loss += loss/len(train_loader)
        # print loss
        #print('Epoch[{}]: {}'.format(epoch, loss))
        print("epoch ", epoch, ": loss = ", train_loss)
        # log
        writer.add_scalar("mse_loss", train_loss, epoch)

        # save model
        print('=====>Save model ...')
        save_dir = 'model_dir'
        save_model(net, os.path.join(save_dir, 'model_locomotion{}.pth.tar'.format(epoch)))
