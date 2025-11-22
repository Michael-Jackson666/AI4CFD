import torch
import torch.nn as nn
import torch.optim as optim

from quadrature import *
from integration import *
from tnn import *

import os

torch.set_default_tensor_type(torch.DoubleTensor)

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
dim = 5
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 100
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
N = len(x)

# ********** create a neural network model **********
p = 50
sizes = [1, 100, 100, 100, p]
# activation = TNN_Tanh
activation = TNN_Sin
# activation = TNN_ReQU
# activation = TNN_Sigmoid

# define forced boundary condition function.
def bd(x):
    return (x-a)*(b-x)

# define derivative of forced boundary condition function.
def grad_bd(x):
    return -2*x+a+b

def grad_grad_bd(x):
    return -2*torch.ones_like(x)


model = TNN(dim,sizes,activation,bd=bd,grad_bd=grad_bd,grad_grad_bd=grad_grad_bd,scaling=False).to(dtype).to(device)
print(model)

# ********** define loss function **********
def criterion(model, w, x):
    phi, grad_phi, grad_grad_phi = model(w,x,need_grad=2)
    alpha = torch.ones(p,device=device,dtype=dtype)

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    A = part1
    M = part0

    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L,A.t()).t()
    D = torch.linalg.solve(L,C)
    E, V = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    u = torch.linalg.solve(L.t(),V[:,ind])


    # laplace operator
    phi_expand = phi.expand(dim,-1,-1,-1).clone()
    phi_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi
    grad_grad_phi_new = phi_expand.transpose(0,1).flatten(1,2)

    alpha_new = u.repeat(dim)

    part0 = Int2TNN(w, u, phi, u, phi)
    
    part1 = Int2TNN(w, alpha_new, grad_grad_phi_new, alpha_new, grad_grad_phi_new)

    part2 = Int2TNN(w, alpha_new, grad_grad_phi_new, u, phi)

    loss = torch.sqrt(part1+lam**2*part0+2*lam*part2)

    return loss


# ********** post_process **********
# exact eigenvalue
exactlam = dim*pi**2
# exact eigenfunction
F = torch.ones((dim,1,N),device=device,dtype=dtype)
F = torch.sin(pi*x)*F
alpha_F = torch.ones(1,device=device,dtype=dtype)
# gradient of exact eigenfunction
grad_F = torch.ones((dim,1,N),device=device,dtype=dtype)
grad_F = pi*torch.cos(pi*x)*grad_F


def post_process(model, w, x):
    phi, grad_phi, grad_grad_phi = model(w,x,need_grad=2)
    alpha = torch.ones(p,device=device,dtype=dtype)

    part0 = Int2TNN(w, alpha, phi, alpha, phi, if_sum=False)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    A = part1
    M = part0

    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L,A.t()).t()
    D = torch.linalg.solve(L,C)
    E, V = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    u = torch.linalg.solve(L.t(),V[:,ind])

    # compute errors
    errorlam = (lam - exactlam) / exactlam
    error0 = error0_estimate(w, alpha_F, F, u, phi, projection=True)
    error1 = error1_estimate(w, alpha_F, F, u, phi, grad_F, grad_phi, projection=True)
    print('{:<9}{:<25}'.format('errorE = ', errorlam))
    print('{:<9}{:<25}'.format('error0 = ', error0))
    print('{:<9}{:<25}'.format('error1 = ', error1))

    return


# ********** training process **********
# parameters
lr = 0.003
epochs = 50000
print_every = 100
save = False
# optimizer used
optimizer = optim.Adam(model.parameters(), lr=lr)

# training
for e in range(epochs):
    loss = criterion(model, w, x)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, x)
        # save model
        if save:
            if not os.path.exists('model'): os.mkdir('model')
            torch.save(model, 'model/model{}.pkl'.format(e))
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
        post_process(model, w, x)
        # save model
        if save:
            torch.save(model, 'model/model{}.pkl'.format(e+1))
print('*'*40)
print('Done!')


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
