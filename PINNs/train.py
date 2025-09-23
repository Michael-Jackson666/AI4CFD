"""
Training script for Physics-Informed Neural Networks (PINNs).
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import create_pinn_model
from pde_definitions import create_pde
from utils import generate_1d_poisson_data, generate_2d_poisson_data, plot_1d_solution, plot_2d_comparison


class PINNTrainer:
    """Trainer class for Physics-Informed Neural Networks."""
    
    def __init__(self, model, pde, device='cpu'):
        self.model = model.to(device)
        self.pde = pde
        self.device = device
        self.history = {'loss': [], 'pde_loss': [], 'bc_loss': [], 'data_loss': []}
    
    def create_training_data(self, problem_type, **kwargs):
        """Create training data based on problem type."""
        if problem_type == 'poisson_1d':
            x, u_exact, f = generate_1d_poisson_data(**kwargs)
            # Create collocation points
            x_collocation = np.random.uniform(-1, 1, (1000, 1))
            # Boundary points
            x_bc = np.array([[-1.0], [1.0]])
            u_bc = np.zeros((2, 1))  # Homogeneous Dirichlet BC
            
            return {
                'x_collocation': torch.FloatTensor(x_collocation).to(self.device),
                'x_bc': torch.FloatTensor(x_bc).to(self.device),
                'u_bc': torch.FloatTensor(u_bc).to(self.device),
                'x_data': torch.FloatTensor(x).to(self.device) if kwargs.get('use_data', False) else None,
                'u_data': torch.FloatTensor(u_exact.reshape(-1, 1)).to(self.device) if u_exact is not None and kwargs.get('use_data', False) else None
            }
        
        elif problem_type == 'poisson_2d':
            coords, u_exact, f = generate_2d_poisson_data(**kwargs)
            # Create collocation points
            x_collocation = np.random.uniform(-1, 1, (2000, 2))
            # Boundary points (simplified)
            n_bc = 200
            x_bc = []
            u_bc = []
            
            # Four boundaries
            x_bc.append(np.column_stack([np.full(n_bc//4, -1), np.linspace(-1, 1, n_bc//4)]))  # Left
            x_bc.append(np.column_stack([np.full(n_bc//4, 1), np.linspace(-1, 1, n_bc//4)]))   # Right
            x_bc.append(np.column_stack([np.linspace(-1, 1, n_bc//4), np.full(n_bc//4, -1)]))  # Bottom
            x_bc.append(np.column_stack([np.linspace(-1, 1, n_bc//4), np.full(n_bc//4, 1)]))   # Top
            
            x_bc = np.vstack(x_bc)
            u_bc = np.zeros((x_bc.shape[0], 1))
            
            return {
                'x_collocation': torch.FloatTensor(x_collocation).to(self.device),
                'x_bc': torch.FloatTensor(x_bc).to(self.device),
                'u_bc': torch.FloatTensor(u_bc).to(self.device),
                'x_data': torch.FloatTensor(coords).to(self.device) if kwargs.get('use_data', False) else None,
                'u_data': torch.FloatTensor(u_exact.reshape(-1, 1)).to(self.device) if u_exact is not None and kwargs.get('use_data', False) else None
            }
        
        elif problem_type == 'burgers':
            # Time-dependent problem
            nx, nt = kwargs.get('nx', 256), kwargs.get('nt', 100)
            x = np.linspace(-1, 1, nx)
            t = np.linspace(0, 1, nt)
            
            # Collocation points in space-time
            x_col = np.random.uniform(-1, 1, (2000, 1))
            t_col = np.random.uniform(0, 1, (2000, 1))
            xt_collocation = np.hstack([x_col, t_col])
            
            # Initial condition: u(x, 0) = -sin(πx)
            x_ic = np.linspace(-1, 1, 100).reshape(-1, 1)
            t_ic = np.zeros_like(x_ic)
            xt_ic = np.hstack([x_ic, t_ic])
            u_ic = -np.sin(np.pi * x_ic)
            
            # Boundary conditions: u(-1, t) = u(1, t) = 0
            t_bc = np.linspace(0, 1, 100).reshape(-1, 1)
            x_bc_left = np.full_like(t_bc, -1)
            x_bc_right = np.full_like(t_bc, 1)
            xt_bc = np.vstack([np.hstack([x_bc_left, t_bc]), np.hstack([x_bc_right, t_bc])])
            u_bc = np.zeros((xt_bc.shape[0], 1))
            
            return {
                'xt_collocation': torch.FloatTensor(xt_collocation).to(self.device),
                'xt_ic': torch.FloatTensor(xt_ic).to(self.device),
                'u_ic': torch.FloatTensor(u_ic).to(self.device),
                'xt_bc': torch.FloatTensor(xt_bc).to(self.device),
                'u_bc': torch.FloatTensor(u_bc).to(self.device)
            }
        
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def compute_loss(self, data, lambda_pde=1.0, lambda_bc=10.0, lambda_data=1.0):
        """Compute total loss including PDE, boundary, and data terms."""
        total_loss = 0.0
        loss_dict = {}
        
        # PDE residual loss
        if 'x_collocation' in data:
            x_col = data['x_collocation']
            x_col.requires_grad_(True)
            u_pred = self.model(x_col)
            pde_residual = self.pde.residual(x_col, u_pred)
            pde_loss = torch.mean(pde_residual**2)
            total_loss += lambda_pde * pde_loss
            loss_dict['pde_loss'] = pde_loss.item()
        
        elif 'xt_collocation' in data:  # Time-dependent problems
            xt_col = data['xt_collocation']
            xt_col.requires_grad_(True)
            u_pred = self.model(xt_col)
            pde_residual = self.pde.residual(xt_col, u_pred)
            pde_loss = torch.mean(pde_residual**2)
            total_loss += lambda_pde * pde_loss
            loss_dict['pde_loss'] = pde_loss.item()
        
        # Boundary condition loss
        if 'x_bc' in data:
            x_bc = data['x_bc']
            u_bc_pred = self.model(x_bc)
            u_bc_target = data['u_bc']
            bc_loss = torch.mean((u_bc_pred - u_bc_target)**2)
            total_loss += lambda_bc * bc_loss
            loss_dict['bc_loss'] = bc_loss.item()
        
        elif 'xt_bc' in data:  # Time-dependent boundary conditions
            xt_bc = data['xt_bc']
            u_bc_pred = self.model(xt_bc)
            u_bc_target = data['u_bc']
            bc_loss = torch.mean((u_bc_pred - u_bc_target)**2)
            total_loss += lambda_bc * bc_loss
            loss_dict['bc_loss'] = bc_loss.item()
        
        # Initial condition loss (for time-dependent problems)
        if 'xt_ic' in data:
            xt_ic = data['xt_ic']
            u_ic_pred = self.model(xt_ic)
            u_ic_target = data['u_ic']
            ic_loss = torch.mean((u_ic_pred - u_ic_target)**2)
            total_loss += lambda_bc * ic_loss
            loss_dict['ic_loss'] = ic_loss.item()
        
        # Data loss (if training data is available)
        if data.get('x_data') is not None and data.get('u_data') is not None:
            x_data = data['x_data']
            u_data_pred = self.model(x_data)
            u_data_target = data['u_data']
            data_loss = torch.mean((u_data_pred - u_data_target)**2)
            total_loss += lambda_data * data_loss
            loss_dict['data_loss'] = data_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def train(self, data, epochs=10000, lr=0.001, optimizer_type='adam', 
              lambda_pde=1.0, lambda_bc=10.0, lambda_data=1.0, 
              print_every=1000, save_dir=None):
        """Train the PINN model."""
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS(self.model.parameters(), lr=lr, 
                                   max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            
            if optimizer_type.lower() == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    loss, loss_dict = self.compute_loss(data, lambda_pde, lambda_bc, lambda_data)
                    loss.backward()
                    return loss
                
                optimizer.step(closure)
                loss, loss_dict = self.compute_loss(data, lambda_pde, lambda_bc, lambda_data)
                
            else:  # Adam or other optimizers
                optimizer.zero_grad()
                loss, loss_dict = self.compute_loss(data, lambda_pde, lambda_bc, lambda_data)
                loss.backward()
                optimizer.step()
            
            # Record history
            self.history['loss'].append(loss_dict['total_loss'])
            self.history['pde_loss'].append(loss_dict.get('pde_loss', 0))
            self.history['bc_loss'].append(loss_dict.get('bc_loss', 0))
            self.history['data_loss'].append(loss_dict.get('data_loss', 0))
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:6d}: Loss = {loss_dict['total_loss']:.6e}, "
                      f"PDE = {loss_dict.get('pde_loss', 0):.6e}, "
                      f"BC = {loss_dict.get('bc_loss', 0):.6e}, "
                      f"Time = {elapsed:.2f}s")
        
        # Save model if directory specified
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))
            torch.save(self.history, os.path.join(save_dir, 'history.pth'))
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
    
    def evaluate(self, test_data):
        """Evaluate the trained model."""
        self.model.eval()
        with torch.no_grad():
            if 'x_test' in test_data:
                x_test = test_data['x_test']
                u_pred = self.model(x_test)
                
                if 'u_exact' in test_data:
                    u_exact = test_data['u_exact']
                    error = torch.mean((u_pred - u_exact)**2).item()
                    rel_error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
                    print(f"Test MSE: {error:.6e}")
                    print(f"Relative L2 error: {rel_error:.6e}")
                    return u_pred, error, rel_error.item()
                else:
                    return u_pred
        
        self.model.train()


def main():
    parser = argparse.ArgumentParser(description='Train Physics-Informed Neural Networks')
    parser.add_argument('--problem', type=str, default='poisson_1d', 
                       choices=['poisson_1d', 'poisson_2d', 'burgers', 'heat', 'wave'],
                       help='Type of PDE problem to solve')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'fourier', 'resnet', 'multiscale', 'adaptive'],
                       help='Type of neural network architecture')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'lbfgs'],
                       help='Optimizer type')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--lambda_pde', type=float, default=1.0, help='PDE loss weight')
    parser.add_argument('--lambda_bc', type=float, default=10.0, help='Boundary condition loss weight')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Solving {args.problem} problem with {args.model_type} model")
    
    # Create model
    if args.problem.startswith('poisson_1d'):
        input_dim = 1
    elif args.problem.startswith('poisson_2d'):
        input_dim = 2
    else:  # Time-dependent problems
        input_dim = 2  # (x, t)
    
    model = create_pinn_model(
        model_type=args.model_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers
    )
    
    # Create PDE
    if args.problem.startswith('poisson'):
        pde = create_pde('poisson', dim=1 if args.problem.endswith('1d') else 2)
    elif args.problem == 'burgers':
        pde = create_pde('burgers', nu=0.01)
    elif args.problem == 'heat':
        pde = create_pde('heat', alpha=1.0)
    elif args.problem == 'wave':
        pde = create_pde('wave', c=1.0)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")
    
    # Create trainer
    trainer = PINNTrainer(model, pde, device)
    
    # Create training data
    training_data = trainer.create_training_data(args.problem)
    
    # Train model
    trainer.train(
        training_data,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_type=args.optimizer,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        save_dir=args.save_dir
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(trainer.history['loss'], label='Total Loss')
    plt.semilogy(trainer.history['pde_loss'], label='PDE Loss')
    plt.semilogy(trainer.history['bc_loss'], label='BC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.grid(True)
    
    # Evaluate and plot solution
    if args.problem == 'poisson_1d':
        x_test = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
        u_pred = model(x_test)
        u_exact = torch.sin(torch.pi * x_test)
        
        plt.subplot(1, 2, 2)
        plt.plot(x_test.cpu().numpy(), u_pred.detach().cpu().numpy(), 'b-', label='Predicted', linewidth=2)
        plt.plot(x_test.cpu().numpy(), u_exact.cpu().numpy(), 'r--', label='Exact', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.title('Solution Comparison')
        plt.grid(True)
        
        # Compute error
        error = torch.mean((u_pred - u_exact)**2).item()
        rel_error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
        print(f"Test MSE: {error:.6e}")
        print(f"Relative L2 error: {rel_error:.6e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()