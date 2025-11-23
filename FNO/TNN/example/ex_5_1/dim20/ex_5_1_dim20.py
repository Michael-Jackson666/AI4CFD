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
a = -1
b = 1
dim = 20
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 200
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
N = len(x)
# print(w)
print(N)

# ********** create a neural network model **********
p = 50
size = [1, 100, 100, 100, p]
activation = TNN_Sin

# define forced boundary condition function.
def bd(x):
    return (x-a)*(b-x)

# define derivative of forced boundary condition function.
def grad_bd(x):
    return -2*x+a+b

def grad_grad_bd(x):
    return -2*torch.ones_like(x)


model = TNN(dim,size,activation,bd=bd,grad_bd=grad_bd,grad_grad_bd=grad_grad_bd,scaling=False).to(dtype).to(device)
print(model)

# F(x)=\sum_{k=1}^d\sin(2\pi x_k)\cdot\prod_{i\neq k}^d\sin(\pi x_i)
F = torch.ones((dim,dim,N),dtype=dtype,device=device)
F = torch.sin(pi*x)*F
for i in range(dim):
    F[i,i,:] = torch.sin(2*pi*x)
alpha_F = torch.ones(dim,dtype=dtype,device=device)

grad_F = torch.ones((dim,dim,N),device=device,dtype=dtype)
grad_F = pi*torch.cos(pi*x)*grad_F
for i in range(dim):
    grad_F[i,i,:] = 2*pi*torch.cos(2*pi*x)

# ********** define loss function **********
# loss = \frac{\int|\nabla\Phi(x)|^2dx}{\int\Phi^2(x)dx}
def criterion(model, w, x):
    phi, grad_phi, grad_grad_phi = model(w,x,need_grad=2)
    alpha = torch.ones(p,device=device,dtype=dtype)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = Int2TNN(w, alpha, phi, alpha_F, F, if_sum=False)
    part2 = torch.sum(part2,dim=-1)

    A = part1
    B = (dim+3)*np.pi**2*part2
    C = torch.linalg.solve(A,B)

    # laplace
    phi_expand = phi.expand(dim,-1,-1,-1).clone()
    phi_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi
    grad_grad_phi_new = phi_expand.transpose(0,1).flatten(1,2)
    C_new = C.repeat(dim)


    part1 = Int2TNN(w, C_new, grad_grad_phi_new, C_new, grad_grad_phi_new)

    part2 = Int2TNN(w, alpha_F, F, alpha_F, F)

    part3 = Int2TNN(w, C_new, grad_grad_phi_new, alpha_F, F)

    loss = part1+(dim+3)**2*np.pi**4*part2+2*(dim+3)*np.pi**2*part3

    return loss


# ********** post_process **********
def post_process(model, w, x):
    phi, grad_phi = model(w,x,need_grad=1)
    alpha = torch.ones(p,device=device,dtype=dtype)

    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)

    part2 = Int2TNN(w, alpha, phi, alpha_F, F, if_sum=False)
    part2 = torch.sum(part2,dim=-1)

    A = part1
    B = (dim+3)*np.pi**2*part2
    C = torch.linalg.solve(A,B)

    # compute errors
    error0 = error0_estimate(w, alpha_F, F, C, phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / ((dim+3)*pi**2)
    error1 = error1_estimate(w, alpha_F, F, C, phi, grad_F, grad_phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / ((dim+3)*pi**2)
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

starttime = time.time()
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
    def closure():
        loss = criterion(model, w, x)
        optimizer.zero_grad()
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, x)

        # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model, w, x)