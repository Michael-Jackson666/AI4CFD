""""
1D Poisson equation with Dirichlet boundary conditions
torch backend - fixed numpy/torch compatibility
"""
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x)
    # Convert to appropriate format for computation
    if isinstance(x, np.ndarray):
        return -dy_xx - np.pi ** 2 * np.sin(np.pi * x)
    else:
        return -dy_xx - torch.pi ** 2 * torch.sin(torch.pi * x)

def boundary(x, on_boundary):
    return on_boundary

def func(x):
    # Handle both numpy arrays and torch tensors
    if isinstance(x, np.ndarray):
        return np.sin(np.pi * x)
    else:
        return torch.sin(torch.pi * x)

geom = dde.geometry.Interval(-1, 1)
bc = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Plot PDE residual
x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
