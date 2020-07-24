import sys
import time
import numpy as np
import scipy.io as io
from autograd import grad
import torch

def multiconstraint(*fs): 
    return lambda H, V: (sum(map(lambda f: f(H, V), fs)) / len(fs))

def trajectory(traj):
    
    def trajectory_constraint(H, V):
    
        velocity_scale = 10
        return velocity_scale * np.mean((V[:,-7:-4] - traj)**2)
  
    return trajectory_constraint
    
def foot_sliding(labels):
    
    def foot_sliding_constraint(H, V):
        
        feet = np.array([[12,13,14], [15,16,17],[24,25,26], [27,28,29]])
        contact = (labels > 0.5)
        
        offsets = np.concatenate([
            V[:,feet[:,0:1]],
            np.zeros((V.shape[0],len(feet),1,V.shape[2])),
            V[:,feet[:,2:3]]], axis=2)
        
        def cross(A, B):
            return np.concatenate([
                A[:,:,1:2]*B[:,:,2:3] - A[:,:,2:3]*B[:,:,1:2],
                A[:,:,2:3]*B[:,:,0:1] - A[:,:,0:1]*B[:,:,2:3],
                A[:,:,0:1]*B[:,:,1:2] - A[:,:,1:2]*B[:,:,0:1]
            ], axis=2)
        
        shape = V[:,-5].shape
        rotation = -V[:,-5].reshape(shape[0],1,1,shape[1]) * cross(np.array([[[0,1,0]]]), offsets)
        
        velocity_scale = 10
        shape = V[:,-7,:-1].shape
        cost_feet_x = velocity_scale * np.mean(contact[:,:,:-1] * (((V[:,feet[:,0],1:] - V[:,feet[:,0],:-1]) + V[:,-7,:-1].reshape(shape[0],1,shape[1]) + rotation[:,:,0,:-1])**2))
        shape = V[:,-6,:-1].shape
        cost_feet_z = velocity_scale * np.mean(contact[:,:,:-1] * (((V[:,feet[:,2],1:] - V[:,feet[:,2],:-1]) + V[:,-6,:-1].reshape(shape[0],1,shape[1]) + rotation[:,:,2,:-1])**2))
        cost_feet_y = velocity_scale * np.mean(contact[:,:,:-1] * ((V[:,feet[:,1],1:] - V[:,feet[:,1],:-1]) **2))
        cost_feet_h = 10.0 * np.mean(np.minimum(V[:,feet[:,1],1:], 0.0)**2)
        
        return (cost_feet_x + cost_feet_z + cost_feet_y + cost_feet_h) / 4
    
    return foot_sliding_constraint

def joint_lengths(
    parents=np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]),
    lengths=np.array([
            2.40,7.15,7.49,2.36,2.37,7.43,7.50,2.41,
            2.04,2.05,1.75,1.76,2.90,4.98,3.48,0.71,
            2.73,5.24,3.44,0.62])):

    def joint_lengths_constraint(H, V):
        J = V[:,:-7].reshape((V.shape[0], len(parents), 3, V.shape[2]))   
        return np.mean((
            np.sqrt(np.sum((J[:,2:] - J[:,parents[2:]])**2, axis=2))
            - lengths[np.newaxis,...,np.newaxis])**2)

    return joint_lengths_constraint
        
    
def constrain(X, autoencoder, preprocess, constraint, alpha=0.1, iterations=100):
    
    input = torch.from_numpy((X - preprocess['Xmean']) / preprocess['Xstd']).to(dtype=torch.float)
  
    def forward(input1):
      x = autoencoder.conv1(input1)
      x = autoencoder.relu1(x)
      x = autoencoder.dropout1(x)
      x = autoencoder.maxpool1(x)[0]
      return x

    def backward(input2):
      x = autoencoder.unpool1d(input2)
      x = autoencoder.dropout2(x)
      x = autoencoder.conv2(x)
      return x
    
    H = forward(input)
    V = (backward(H).cpu().numpy() * preprocess['Xstd']) + preprocess['Xmean']
    
    Hvar = np.copy(H.cpu().numpy())
    Vvar = np.copy(V)

    self_alpha = alpha
    self_beta1 = 0.9
    self_beta2 = 0.999
    self_eps = 1e-05
    self_batchsize = 1

    self_params = [Hvar]
    self_m0params = [np.zeros(p.shape) for p in self_params]
    self_m1params = [np.zeros(p.shape) for p in self_params]
    self_t = np.array([1])

    for i in range(iterations):
        #derivative with first argument
        cost = constraint(self_params[0],Vvar)
        # print('Constraint Iteration %i, error %f' % (i, cost))
        grad_cost = grad(constraint,0)
        gparams = grad_cost(self_params,Vvar)
        m0params = [self_beta1 * m0p + (1-self_beta1) *  gp     for m0p, gp in zip(self_m0params, gparams)]
        m1params = [self_beta2 * m1p + (1-self_beta2) * (gp*gp) for m1p, gp in zip(self_m1params, gparams)]
        params = [p - (self_alpha / self_batchsize) * 
                  ((m0p/(1-(self_beta1**self_t[0]))) /
            (np.sqrt(m1p/(1-(self_beta2**self_t[0]))) + self_eps))
            for p, m0p, m1p in zip(self_params, m0params, m1params)]
        self_params = params
        self_m0params = m0params
        self_m1params = m1params
        self_t += 1
        start = time.clock()

    print('Constraint: %0.4f' % (time.clock() - start))
    # constraint_func = torch.function([], cost, updates=updates)

    
    # for i in range(iterations):
    #    cost = constraint_func()
    #    print('Constraint Iteration %i, error %f' % (i, cost))
    # print('Constraint: %0.4f' % (time.clock() - start))
    
    return (np.array(backward(H)) * preprocess['Xstd']) + preprocess['Xmean']
    
    