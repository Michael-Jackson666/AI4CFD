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

name = '0111_5'

# ********** generate data points **********
# computation domain: [a,b]^dim
a = 0
b = 1
dim = 5
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 10
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
N = len(x)
# print(w)
print(N)
# print(torch.sum(w))


# -\Delta u=\frac{\pi^2}{4} F(x)
#         u=F(x)
# where F(x)=\sum_{i=1}^d\sin(\pi x_i/2)
F = torch.ones((dim,dim,N),dtype=dtype,device=device)
for i in range(dim):
    F[i,i,:] = torch.sin(pi/2*x)
alpha_F = torch.ones(dim,dtype=dtype,device=device)

# gradient of exact eigenfunction
grad_F = torch.zeros((dim,dim,N),device=device,dtype=dtype)
for i in range(dim):
    grad_F[i,i,:] = pi/2*torch.cos(pi/2*x)

# value of F(x) on \partial\Omega
F_bd0 = torch.ones((dim,dim,N),dtype=dtype,device=device)
for i in range(dim):
    # F_bd0[i,i,:] = -torch.sin(pi/2*torch.ones_like(x))
    F_bd0[i,i,:] = torch.sin(pi/2*torch.zeros_like(x))


F_bd1 = torch.ones((dim,dim,N),dtype=dtype,device=device)
for i in range(dim):
    F_bd1[i,i,:] = torch.sin(pi/2*torch.ones_like(x))



# define forced boundary condition function.
def bd(x):
    return (x-a)*(b-x)

# define derivative of forced boundary condition function.
def grad_bd(x):
    return -2*x+a+b

def grad_grad_bd(x):
    return -2*torch.ones_like(x)


# ********** create a neural network model **********
p_b = 20
p_0 = 20
size_b = [1, 50, 50, 50, p_b]
size_0 = [1, 50, 50, 50, p_0]
# activation = TNN_Sin
# activation_b = TNN_Tanh
# activation_0 = TNN_Tanh

activation_b = TNN_Sin
activation_0 = TNN_Sin

# TNN for bd condition on \partial\Omega
model_b = TNN(dim,size_b,activation_b,bd=None,grad_bd=None,scaling=True).to(dtype).to(device)
print(model_b)
# TNN for solution in \Omega
model_0 = TNN(dim,size_0,activation_0,bd=bd,grad_bd=grad_bd,grad_grad_bd=grad_grad_bd,scaling=False).to(dtype).to(device)
print(model_0)


# *************** loss function for bd condition ************
def criterion_b(model_b, w, x):
    alpha = torch.ones(p_b,device=device,dtype=dtype)
    # alpha = model_b.scaling_par()
    phi = model_b(w,x,need_grad=0,normed=False)
    # compute norm
    norm = torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)

    phi = phi / norm

    # bd value of each \phi_{i,j}
    phi_bd0 = model_b(w,torch.zeros_like(x),need_grad=0,normed=False)
    # phi_bd0 = model_b(w,-torch.ones_like(x),need_grad=0,normed=False)
    phi_bd0 = phi_bd0 / norm

    phi_bd1 = model_b(w,torch.ones_like(x),need_grad=0,normed=False)
    phi_bd1 = phi_bd1 / norm


    # 
    A = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, phi_bd0, phi_bd0, if_sum=False)\
        +Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, phi_bd1, phi_bd1, if_sum=False)

    B = torch.sum(Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, F_bd0, phi_bd0, if_sum=False),dim=0)\
        +torch.sum(Int2TNN_amend_1d(w, w, alpha_F, F, alpha, phi, F_bd1, phi_bd1, if_sum=False),dim=0)

    alpha = torch.linalg.solve(A,B)

    # print(torch.sum(alpha**2))

    # loss on \partial\Omega
    loss_bd = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, phi_bd0, phi_bd0)\
              +Int2TNN_amend_1d(w, w, alpha_F, F, alpha_F, F, F_bd0, F_bd0)\
              -2*Int2TNN_amend_1d(w, w, alpha, phi, alpha_F, F, phi_bd0, F_bd0)\
              +Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, phi_bd1, phi_bd1)\
              +Int2TNN_amend_1d(w, w, alpha_F, F, alpha_F, F, F_bd1, F_bd1)\
              -2*Int2TNN_amend_1d(w, w, alpha, phi, alpha_F, F, phi_bd1, F_bd1)
    
    loss_bd = torch.sqrt(loss_bd)

    return loss_bd, alpha

print('*'*20,'Training on partial Omega','*'*20)
print('*'*20,'Adam','*'*20)
# ********** training process for bd **********
# parameters
lr = 0.003
epochs = 20000
print_every = 1000
save = True
# optimizer used
optimizer = optim.Adam(model_b.parameters(), lr=lr)

starttime = time.time()
# training
for e in range(epochs):
    loss_bd, alpha_b = criterion_b(model_b, w, x)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss_bd = ', loss_bd.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        # post_process(model, w, x)
        # save model
        # if save:
        #     if not os.path.exists('model'): os.mkdir('model')
        #     torch.save(model, 'model/model{}.pkl'.format(e))
    # optimization process
    optimizer.zero_grad()
    loss_bd.backward()
    optimizer.step()

    # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss_bd = ', loss_bd.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        # post_process(model, w, x)
        # save model
        # torch.save(model, 'model/model{}.pkl'.format(e+1))
    
print('*'*40)
print('Done!')

endtime = time.time()
print('Training took: {}s'.format(endtime - starttime))



print('*'*20,'LBFGS','*'*20)
# ********** training process LBFGS **********
# parametersa
lr = 0.1
epochs = 5000
print_every = 100
save = True
# optimizer used
# optimizer = torch.optim.LBFGS(model_b.parameters(), lr=lr, tolerance_grad=0, tolerance_change=0)
optimizer = torch.optim.LBFGS(model_b.parameters(), lr=lr, tolerance_grad=0, tolerance_change=0)
# training
for e in range(epochs):
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss_bd = ', loss_bd.item()))
        # user-defined post-process
        # post_process(model, w, x)

    def closure():
        loss_bd, _ = criterion_b(model_b, w, x)
        optimizer.zero_grad()
        loss_bd.backward()
        return loss_bd
    loss_bd = optimizer.step(closure)

    _, alpha_b = criterion_b(model_b, w, x)


        # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss_bd = ', loss_bd.item()))
        # user-defined post-process
        # post_process(model, w, x)

if not os.path.exists('model'): os.mkdir('model')
torch.save(model_b, './model/{}_model_b.pkl'.format(name))
torch.save(alpha_b, './model/{}_alpha_b.pkl'.format(name))


# fix parameters in model_b
for par in model_b.parameters():
    par.requires_grad_(False)
# alpha_b.detach()
# print(alpha_b)

# ********** define loss function in \Omega **********
# loss = \frac{\int|\nabla\Phi(x)|^2dx}{\int\Phi^2(x)dx}
def criterion_0(model_0, alpha_b, model_b, w, x):
    alpha_0 = torch.ones(p_0,device=device,dtype=dtype)
    phi_0, grad_phi_0, grad_grad_phi_0 = model_0(w,x,need_grad=2)
    phi_b, grad_phi_b, grad_grad_phi_b = model_b(w,x,need_grad=2)

    # print(Int2TNN(w,alpha_b,phi_b,alpha_b,phi_b))

    # Delta u_g
    phi_b_expand = phi_b.expand(dim,-1,-1,-1).clone()
    phi_b_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi_b
    Delta_phi_b = phi_b_expand.transpose(0,1).flatten(1,2)
    alpha_Delta_b = alpha_b.repeat(dim)

    # print("****",Int2TNN(w, alpha_Delta_b, Delta_phi_b, alpha_Delta_b, Delta_phi_b))

    # new right-hand-term
    # alpha_F_b = torch.cat((pi**2/4*alpha_F,alpha_Delta_b),dim=0)
    # F_b = torch.cat((F,Delta_phi_b),dim=1)

    # print(Int2TNN(w, alpha_F, F, alpha_F, F))
    # print(torch.sum(Int2TNN(w, alpha_Delta_b, Delta_phi_b, alpha_0, phi_0, if_sum=False),dim=0)+torch.sum(Int2TNN_amend_1d(w, w, alpha_b, phi_b, alpha_0, phi_0, grad_phi_b, grad_phi_0, if_sum=False),dim=0))

    # 
    A = Int2TNN_amend_1d(w, w, alpha_0, phi_0, alpha_0, phi_0, grad_phi_0, grad_phi_0, if_sum=False)
    B = pi**2/4*torch.sum(Int2TNN(w, alpha_F, F, alpha_0, phi_0, if_sum=False),dim=0)\
        -torch.sum(Int2TNN_amend_1d(w, w, alpha_b, phi_b, alpha_0, phi_0, grad_phi_b, grad_phi_0, if_sum=False),dim=0)
    alpha_0 = torch.linalg.solve(A,B)


    # Delta u_0 operator
    phi_0_expand = phi_0.expand(dim,-1,-1,-1).clone()
    phi_0_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi_0
    Delta_phi_0 = phi_0_expand.transpose(0,1).flatten(1,2)
    alpha_Delta_0 = alpha_0.repeat(dim)


    # Delta u
    Delta_phi = torch.cat((Delta_phi_0,Delta_phi_b),dim=1)
    alpha_Delta = torch.cat((alpha_Delta_0,alpha_Delta_b),dim=0)

    # compute posteriori error estimator.
    Int_F_F = Int2TNN(w, alpha_F, F, alpha_F, F)

    Int_DeltaPhi_DeltaPhi = Int2TNN(w, alpha_Delta, Delta_phi, alpha_Delta, Delta_phi)

    Int_F_DeltaPhi = Int2TNN(w, alpha_F, F, alpha_Delta, Delta_phi)

    # posteriori error estimator
    loss = pi**4/16*Int_F_F + Int_DeltaPhi_DeltaPhi + pi**2/2*Int_F_DeltaPhi
    loss = torch.sqrt(loss)

    return loss



# ********** post_process **********
# exact eigenfunction is F(x)
def post_process(model_0, alpha_b, model_b, w, x):
    alpha_0 = torch.ones(p_0,device=device,dtype=dtype)
    phi_0, grad_phi_0, grad_grad_phi_0 = model_0(w,x,need_grad=2)
    phi_b, grad_phi_b, grad_grad_phi_b = model_b(w,x,need_grad=2)

    # print(Int2TNN(w,alpha_b,phi_b,alpha_b,phi_b))

    # Delta u_g
    phi_b_expand = phi_b.expand(dim,-1,-1,-1).clone()
    phi_b_expand[torch.arange(dim),torch.arange(dim),:,:] = grad_grad_phi_b
    Delta_phi_b = phi_b_expand.transpose(0,1).flatten(1,2)
    alpha_Delta_b = alpha_b.repeat(dim)


    # 
    A = Int2TNN_amend_1d(w, w, alpha_0, phi_0, alpha_0, phi_0, grad_phi_0, grad_phi_0, if_sum=False)
    B = pi**2/4*torch.sum(Int2TNN(w, alpha_F, F, alpha_0, phi_0, if_sum=False),dim=0)\
        -torch.sum(Int2TNN_amend_1d(w, w, alpha_b, phi_b, alpha_0, phi_0, grad_phi_b, grad_phi_0, if_sum=False),dim=0)
    alpha_0 = torch.linalg.solve(A,B)


    alpha = torch.cat((alpha_0,alpha_b),dim=0)
    phi = torch.cat((phi_0,phi_b),dim=1)
    grad_phi = torch.cat((grad_phi_0,grad_phi_b),dim=1)


    # get error
    inner0_phi_phi = Int2TNN(w, alpha, phi, alpha, phi)
    inner0_F_phi = Int2TNN(w, alpha_F, F, alpha, phi)
    inner0_F_F = Int2TNN(w, alpha_F, F, alpha_F, F)


    # compute errors
    error0 = error0_estimate(w, alpha_F, F, alpha, phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / (pi**2/4)
    error1 = error1_estimate(w, alpha_F, F, alpha, phi, grad_F, grad_phi, projection=False) / torch.sqrt(Int2TNN(w, alpha_F, F, alpha_F, F)) / (pi**2/4)
    print('{:<9}{:<25}'.format('error0 = ', error0))
    print('{:<9}{:<25}'.format('error1 = ', error1))
    return


print('*'*20,'Training in Omega','*'*20)
print('*'*20,'Adam','*'*20)
# ********** training process **********
# parameters
lr = 0.003
epochs = 20000
print_every = 1000
save = False
# optimizer used
optimizer = optim.Adam(model_0.parameters(), lr=lr)

starttime = time.time()
# training
for e in range(epochs):
    loss = criterion_0(model_0, alpha_b, model_b, w, x)
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        post_process(model_0, alpha_b, model_b, w, x)
        # save model
        # if save:
        #     if not os.path.exists('model'): os.mkdir('model')
        #     torch.save(model, 'model/model{}.pkl'.format(e))
    # optimization process
    optimizer.zero_grad()
    # loss.backward()
    loss.backward(retain_graph=True)
    optimizer.step()
    # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # print('{:<9}{:}'.format('scaling = ',model.ms['TNN_Scaling'].alpha.data.numpy()))
        # user-defined post-process
        post_process(model_0, alpha_b, model_b, w, x)
        # save model
        # torch.save(model, 'model/model{}.pkl'.format(e+1))
    
print('*'*40)
print('Done!')

endtime = time.time()
print('Training took: {}s'.format(endtime - starttime))



print('*'*20,'LBFGS','*'*20)
# ********** training process LBFGS **********
# parameters
lr = 0.1
epochs = 5000
print_every = 100
save = False
# optimizer used
# optimizer = torch.optim.LBFGS(model_0.parameters(), lr=lr, tolerance_grad=0, tolerance_change=0)
optimizer = torch.optim.LBFGS(model_0.parameters(), lr=lr, tolerance_grad=0, tolerance_change=0)
# training
for e in range(epochs):
    # initial info
    if e == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model_0, alpha_b, model_b, w, x)

    def closure():
        loss = criterion_0(model_0, alpha_b, model_b, w, x)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        return loss
    loss = optimizer.step(closure)

        # post process
    if (e+1) % print_every == 0:
        print('*'*40)
        print('{:<9}{:<25}'.format('epoch = ', e+1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
        # user-defined post-process
        post_process(model_0, alpha_b, model_b, w, x)