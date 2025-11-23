import torch
import torch.nn as nn
import torch.optim as optim

from quadrature import *
from integration import *
from tnn import *

import os
import time

pi = 3.14159265358979323846

# ********** choose data type and device **********
dtype = torch.double
# dtype = torch.float
# device = 'cpu'
device = 'cuda'


# ********** generate data points **********
# computation domain: [a,b]^dim
a = 0
b = 1
dim = 10
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 100
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
N = len(x)
# print(w)
print(N)
# print(torch.sum(w))

# ********** create a neural network model **********
p = 100
size = [1, 100, 100, 100, p]
activation = TNN_Sin

model = TNN(dim,size,activation,bd=None,grad_bd=None,scaling=False).to(dtype).to(device)
print(model)

# F(x)=\sum_{k=1}^d\sin(\pi x_i)
F = torch.ones((dim,dim,N),dtype=dtype,device=device)
for i in range(dim):
    F[i,i,:] = torch.sin(pi*x)
alpha_F = torch.ones(dim,dtype=dtype,device=device)

# gradient of exact eigenfunction
grad_F = torch.zeros((dim,dim,N),device=device,dtype=dtype)
for i in range(dim):
    grad_F[i,i,:] = pi*torch.cos(pi*x)


# 
grad_F0 = torch.zeros((dim,dim,N),device=device,dtype=dtype)
for i in range(dim):
    grad_F0[i,i,:] = -pi*torch.ones_like(x)


grad_F1 = torch.zeros((dim,dim,N),device=device,dtype=dtype)
for i in range(dim):
    grad_F1[i,i,:] = -pi*torch.ones_like(x)


# ********** define loss function **********
# loss = \frac{\int|\nabla\Phi(x)|^2dx}{\int\Phi^2(x)dx}
def criterion(model, w, x):
    phi, grad_phi, grad_grad_phi = model(w,x,need_grad=2,normed=False)
    norm = torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)

    phi = phi / norm
    grad_phi = grad_phi / norm
    grad_grad_phi = grad_grad_phi / norm

    # value on boundary
    phi0, grad_phi0 = model(w,torch.zeros_like(x),need_grad=1,normed=False)
    grad_phi0 = -grad_phi0 / norm
    phi0 = phi0 / norm

    phi1, grad_phi1 = model(w,torch.ones_like(x),need_grad=1,normed=False)
    grad_phi1 = grad_phi1 / norm
    phi1 = phi1 / norm


    alpha = torch.ones(p,device=device,dtype=dtype)

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = Int2TNN(w, alpha_F, F, alpha, phi, if_sum=False)
    part2 = torch.sum(part2,dim=0)

    part3 = Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, grad_F0, phi0, if_sum=False)\
            +Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, grad_F1, phi1, if_sum=False)
    part3 = torch.sum(part3,dim=0)

    A = part1 + pi**2*part0
    B = 2*pi**2*part2 + part3
    C = torch.linalg.solve(A,B)


    # laplace operator
    phi_expand = phi.expand(dim,-1,-1,-1).clone()
    phi_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi
    Delta_phi = phi_expand.transpose(0,1).flatten(1,2)
    alpha_Delta = C.repeat(dim)


    # compute posteriori error estimator.
    Int_F_F = Int2TNN(w, alpha_F, F, alpha_F, F)

    Int_Phi_Phi = Int2TNN(w, C, phi, C, phi)

    Int_DeltaPhi_DeltaPhi = Int2TNN(w, alpha_Delta, Delta_phi, alpha_Delta, Delta_phi)

    Int_F_Phi = Int2TNN(w, alpha_F, F, C, phi)

    Int_F_DeltaPhi = Int2TNN(w, alpha_F, F, alpha_Delta, Delta_phi)

    Int_Phi_DeltaPhi = Int2TNN(w, C, phi, alpha_Delta, Delta_phi)

    # on \partial\Omega
    nabla_phi_cdot_n0_part1 = Int2TNN_amend_1d(w, w, C, phi, C, phi, grad_phi0, grad_phi0)
    nabla_phi_cdot_n0_part2 = Int2TNN_amend_1d(w, w, C, phi, alpha_F, F, grad_phi0, grad_F0)
    nabla_phi_cdot_n0_part3 = Int2TNN_amend_1d(w, w, alpha_F, F, alpha_F, F, grad_F0, grad_F0)

    # print(Int2TNN_amend_1d(w, w, C, phi, alpha_F, F, grad_phi0, grad_F0)-Int2TNN_amend_1d(w, w, alpha_F, F, C, phi, grad_F0, grad_phi0))

    nabla_phi_cdot_n1_part1 = Int2TNN_amend_1d(w, w, C, phi, C, phi, grad_phi1, grad_phi1)
    nabla_phi_cdot_n1_part2 = Int2TNN_amend_1d(w, w, C, phi, alpha_F, F, grad_phi1, grad_F1)
    nabla_phi_cdot_n1_part3 = Int2TNN_amend_1d(w, w, alpha_F, F, alpha_F, F, grad_F1, grad_F1)

    loss_bd = nabla_phi_cdot_n0_part1-2*nabla_phi_cdot_n0_part2+nabla_phi_cdot_n0_part3\
              +nabla_phi_cdot_n1_part1-2*nabla_phi_cdot_n1_part2+nabla_phi_cdot_n1_part3

    # posteriori error estimator
    loss = 4*pi**2*Int_F_F + pi**2*Int_Phi_Phi + 1 / pi**2 * Int_DeltaPhi_DeltaPhi\
            -4*pi**2*Int_F_Phi+4*Int_F_DeltaPhi-2*Int_Phi_DeltaPhi\
            +loss_bd


    return loss


# ********** post_process **********
def post_process(model, w, x):
    phi, grad_phi, grad_grad_phi = model(w,x,need_grad=2,normed=False)
    norm = torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)

    phi = phi / norm
    grad_phi = grad_phi / norm
    grad_grad_phi = grad_grad_phi / norm

    # value on boundary
    phi0, grad_phi0 = model(w,torch.zeros_like(x),need_grad=1,normed=False)
    grad_phi0 = -grad_phi0 / norm
    phi0 = phi0 / norm

    phi1, grad_phi1 = model(w,torch.ones_like(x),need_grad=1,normed=False)
    grad_phi1 = grad_phi1 / norm
    phi1 = phi1 / norm


    alpha = torch.ones(p,device=device,dtype=dtype)

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = Int2TNN(w, alpha_F, F, alpha, phi, if_sum=False)
    part2 = torch.sum(part2,dim=0)

    part3 = Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, grad_F0, phi0, if_sum=False)\
            +Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, grad_F1, phi1, if_sum=False)
    part3 = torch.sum(part3,dim=0)


    A = part1 + pi**2*part0
    B = 2*pi**2*part2 + part3
    C = torch.linalg.solve(A,B)

    # compute errors
    error0 = error0_estimate(w, alpha_F, F, C, phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / (2*pi**2)
    error1 = error1_estimate(w, alpha_F, F, C, phi, grad_F, grad_phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / (2*pi**2)
    print('{:<9}{:<25}'.format('error0 = ', error0))
    print('{:<9}{:<25}'.format('error1 = ', error1))
    return


# ********** training process **********
# parameters
lr = 0.003
epochs = 50000
print_every = 100
save = True
# optimizer used
optimizer = optim.Adam(model.parameters(), lr=lr)

starttime = time.time()
# training
for e in range(epochs):
    loss = criterion(model, w, x)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        post_process(model, w, x)
        # save model
        # if save:
        #     if not os.path.exists('model'): os.mkdir('model')
        #     torch.save(model, 'model/model{}.pkl'.format(e))
    # optimization process
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        post_process(model, w, x)
        # save model
        # torch.save(model, 'model/model{}.pkl'.format(e+1))

print('*'*40)
print('Done!')

endtime = time.time()
print('Training took: {}s'.format(endtime - starttime))



print('*'*20,'LBFGS','*'*20)
# ********** training process LBFGS **********
# parameters
lr = 1
epochs = 10000
print_every = 100
save = True
# optimizer used
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
# training
for e in range(epochs):
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, x)

    def closure():
        loss = criterion(model, w, x)
        optimizer.zero_grad()
        loss.backward()
        return loss
    loss = optimizer.step(closure)

        # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, x)