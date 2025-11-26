import torch
import torch.nn as nn
import torch.optim as optim

from quadrature import *
from integration import *
from tnn import *

import os
import time
import numpy as np
import matplotlib.pyplot as plt

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
dim = 5
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

# Store loss history for plotting
loss_history_adam = []
epoch_history_adam = []

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
    
    # Record loss every iteration
    loss_history_adam.append(loss.item())
    epoch_history_adam.append(e)
    
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


# ********** Visualization: 2D slices **********
print('*'*40)
print('Generating 2D slice visualizations...')

# Fix dimensions 3, 4, 5 at 0, visualize dimensions 1 and 2
n_plot = 50
x1 = torch.linspace(a, b, n_plot, dtype=dtype, device=device)
x2 = torch.linspace(a, b, n_plot, dtype=dtype, device=device)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

# Create 5D points with fixed values for dims 3,4,5
X_plot = torch.zeros(n_plot**2, dim, dtype=dtype, device=device)
X_plot[:, 0] = X1.flatten()
X_plot[:, 1] = X2.flatten()
X_plot[:, 2] = 0.0  # Fixed at 0
X_plot[:, 3] = 0.0  # Fixed at 0
X_plot[:, 4] = 0.0  # Fixed at 0

# Compute solution and exact solution
with torch.no_grad():
    phi, grad_phi = model(w, x, need_grad=1)
    alpha = torch.ones(p, device=device, dtype=dtype)
    
    part1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False)
    part2 = Int2TNN(w, alpha, phi, alpha_F, F, if_sum=False)
    part2 = torch.sum(part2, dim=-1)
    A = part1
    B = (dim+3)*np.pi**2*part2
    C = torch.linalg.solve(A, B)
    
    # Evaluate on plot points
    phi_plot, _ = model(w, X_plot, need_grad=1)
    u_pred = torch.zeros(n_plot**2, dtype=dtype, device=device)
    for i in range(p):
        prod = torch.ones(n_plot**2, dtype=dtype, device=device)
        for d in range(dim):
            prod = prod * phi_plot[d, i, :]
        u_pred = u_pred + C[i] * prod
    
    # Exact solution at slice
    u_exact = torch.zeros(n_plot**2, dtype=dtype, device=device)
    for i in range(dim):
        term = torch.sin(2*pi*X_plot[:, i]) if i < 2 else torch.sin(pi*X_plot[:, i])
        for j in range(dim):
            if j != i:
                term = term * torch.sin(pi*X_plot[:, j])
        u_exact = u_exact + term

u_pred = u_pred.reshape(n_plot, n_plot).cpu().numpy()
u_exact = u_exact.reshape(n_plot, n_plot).cpu().numpy()
X1_cpu = X1.cpu().numpy()
X2_cpu = X2.cpu().numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Predicted solution
im0 = axes[0].contourf(X1_cpu, X2_cpu, u_pred, levels=20, cmap='RdBu_r')
axes[0].set_title('TNN Prediction (x3=x4=x5=0)', fontsize=14)
axes[0].set_xlabel('$x_1$', fontsize=12)
axes[0].set_ylabel('$x_2$', fontsize=12)
plt.colorbar(im0, ax=axes[0])

# Exact solution
im1 = axes[1].contourf(X1_cpu, X2_cpu, u_exact, levels=20, cmap='RdBu_r')
axes[1].set_title('Exact Solution (x3=x4=x5=0)', fontsize=14)
axes[1].set_xlabel('$x_1$', fontsize=12)
axes[1].set_ylabel('$x_2$', fontsize=12)
plt.colorbar(im1, ax=axes[1])

# Error
error = np.abs(u_pred - u_exact)
im2 = axes[2].contourf(X1_cpu, X2_cpu, error, levels=20, cmap='hot')
axes[2].set_title('Absolute Error', fontsize=14)
axes[2].set_xlabel('$x_1$', fontsize=12)
axes[2].set_ylabel('$x_2$', fontsize=12)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('solution_2d_slice.png', dpi=150, bbox_inches='tight')
print('Saved: solution_2d_slice.png')
plt.show()

print('Visualization complete!')