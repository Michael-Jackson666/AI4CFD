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


dim = 5
# ********** generate data points **********
z, w = Hermite_Gauss_Quad(200,device=device, dtype=dtype, modified=False)
N = len(z)
# print(w)
print(N)

# ********** create a neural network model **********
p = 50
sizes = [1, 100, 100, 100, p]
activation = TNN_Sin


model = TNN(dim,sizes,activation,bd=None,grad_bd=None,grad_grad_bd=None,scaling=False).to(dtype).to(device)
print(model)


# V(x)=\sum_{i=1}^dx_i^2
V = torch.ones((dim,dim,N),dtype=dtype,device=device)
for i in range(dim):
    V[i,i,:] = z**2
# print(V)
alpha_V = torch.ones(dim,dtype=dtype,device=device)

def criterion_ritz(model, w, z):
    phi, grad_phi = model(w,z,need_grad=1)
    alpha = torch.ones(p,device=device,dtype=dtype)

    grad_phi = grad_phi - z*phi

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = torch.sum(Int3TNN(w, alpha_V, V, alpha, phi, alpha, phi, if_sum=False),dim=0)

    A = part1+part2
    M = part0

    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L,A.t()).t()
    D = torch.linalg.solve(L,C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    loss = E[ind]

    return loss


def criterion_eta(model, w, z):
    phi, grad_phi, grad_grad_phi = model(w,z,need_grad=2)
    alpha = torch.ones(p,device=device,dtype=dtype)

    grad_grad_phi = grad_grad_phi-2*z*grad_phi+(z**2-1)*phi
    grad_phi = grad_phi - z*phi

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = torch.sum(Int3TNN(w, alpha_V, V, alpha, phi, alpha, phi, if_sum=False),dim=0)


    A = part1+part2
    M = part0

    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L,A.t()).t()
    D = torch.linalg.solve(L,C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    u = torch.linalg.solve(L.t(),U[:,ind])

    # laplace operator
    phi_expand = phi.expand(dim,-1,-1,-1).clone()
    phi_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi
    grad_grad_phi_new = phi_expand.transpose(0,1).flatten(1,2)
    alpha_new = u.repeat(dim)


    part0 = Int2TNN(w, u, phi, u, phi)

    part1 = Int2TNN(w, alpha_new, grad_grad_phi_new, alpha_new, grad_grad_phi_new)

    part2 = Int4TNN(w, alpha_V, V, u, phi, alpha_V, V, u, phi)

    part3 = Int2TNN(w, alpha_new, grad_grad_phi_new, u, phi)

    part4 = Int3TNN(w, alpha_V, V, u, phi, alpha_new, grad_grad_phi_new)

    part5 = Int3TNN(w, alpha_V, V, u, phi, u, phi)

    loss = torch.sqrt(lam**2*part0+part1+part2+2*lam*part3-2*part4-2*lam*part5)

    return loss


# ********** post_process **********
# exact eigenvalue
exactlam = dim
# exact eigenfunction
F = torch.ones((dim,1,N),device=device,dtype=dtype)
alpha_F = torch.ones(1,device=device,dtype=dtype)
# gradient of exact eigenfunction
grad_F = torch.ones((dim,1,N),device=device,dtype=dtype)
grad_F = -z*grad_F


def post_process(model, w, z):
    phi, grad_phi, grad_grad_phi = model(w,z,need_grad=2)
    alpha = torch.ones(p,device=device,dtype=dtype)

    grad_grad_phi = grad_grad_phi-2*z*grad_phi+(z**2-1)*phi
    grad_phi = grad_phi - z*phi

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = torch.sum(Int3TNN(w, alpha_V, V, alpha, phi, alpha, phi, if_sum=False),dim=0)

    A = part1+part2
    M = part0

    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L,A.t()).t()
    D = torch.linalg.solve(L,C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    u = torch.linalg.solve(L.t(),U[:,ind])

    # compute errors
    errorlam = (lam - exactlam) / exactlam
    error0 = error0_estimate(w, alpha_F, F, u, phi, projection=True)
    error1 = error1_estimate(w, alpha_F, F, u, phi, grad_F, grad_phi, projection=True)
    print('{:<9}{:<25}'.format('errorE = ', errorlam))
    print('{:<9}{:<25}'.format('error0 = ', error0))
    print('{:<9}{:<25}'.format('error1 = ', error1))
    return



# ********** pre-training process **********
# parameters
lr = 0.01
epochs = 10000
print_every = 100
save = True
# optimizer used
optimizer = optim.Adam(model.parameters(), lr=lr)

start_time = time.time()
# training
for e in range(epochs):
    loss = criterion_ritz(model, w, z)
    # loss = criterion_eta(model, w, z)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, z)
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
        # user-defined post-process
        post_process(model, w, z)
        # save model
        # torch.save(model, 'model/model{}.pkl'.format(e+1))
    
print('*'*40)
print('Done!')

end_time = time.time()
print('Training took: {}s'.format(end_time - start_time))



print('*'*20,'LBFGS','*'*20)
# ********** training process LBFGS **********
# parameters
lr = 1
epochs = 10000
print_every = 100
save = True
# optimizer used
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

start_time = time.time()
# training
for e in range(epochs):
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        # print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, z)

    def closure():
        loss = criterion_eta(model, w, z)
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
        post_process(model, w, z)
print('*'*40)
print('Done!')

end_time = time.time()
print('Training took: {}s'.format(end_time - start_time))
