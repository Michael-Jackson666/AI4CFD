"""
Example: Solving 2D Burger's Equation using Fourier Neural Operator (FNO)

This example demonstrates how to use FNO to solve the 2D Burger's equation.
Note: This is a simplified example for demonstration purposes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ai4cfd.fno import FNO2d
from ai4cfd.utils import generate_uniform_mesh_2d

def generate_burgers_data(num_samples=100, grid_size=64, viscosity=0.01):
    """
    Generate synthetic training data for 2D Burger's equation.
    This is a simplified data generation for demonstration.
    """
    # Generate initial conditions (random smooth functions)
    x = torch.linspace(0, 2*np.pi, grid_size)
    y = torch.linspace(0, 2*np.pi, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Input: initial condition + grid coordinates
    input_data = torch.zeros(num_samples, grid_size, grid_size, 3)  # u0, x, y
    output_data = torch.zeros(num_samples, grid_size, grid_size, 1)  # u at t=1
    
    for i in range(num_samples):
        # Random initial condition
        k1, k2 = torch.randint(1, 4, (2,))
        a1, a2 = torch.rand(2) * 2 - 1  # Random coefficients
        
        u0 = a1 * torch.sin(k1 * X) * torch.sin(k1 * Y) + a2 * torch.sin(k2 * X) * torch.sin(k2 * Y)
        
        # Simplified evolution (this would normally require solving the PDE)
        # For demonstration, we use an approximate decay
        u_final = u0 * torch.exp(-viscosity * (k1**2 + k2**2))
        
        input_data[i, :, :, 0] = u0
        input_data[i, :, :, 1] = X
        input_data[i, :, :, 2] = Y
        output_data[i, :, :, 0] = u_final
    
    return input_data, output_data


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate training data
    print("Generating training data...")
    grid_size = 32  # Smaller for faster training
    input_data, output_data = generate_burgers_data(num_samples=1000, grid_size=grid_size)
    
    # Split data
    train_size = int(0.8 * len(input_data))
    val_size = int(0.1 * len(input_data))
    
    train_input = input_data[:train_size].to(device)
    train_output = output_data[:train_size].to(device)
    val_input = input_data[train_size:train_size + val_size].to(device)
    val_output = output_data[train_size:train_size + val_size].to(device)
    test_input = input_data[train_size + val_size:].to(device)
    test_output = output_data[train_size + val_size:].to(device)
    
    # Initialize FNO model
    model = FNO2d(
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4,
        input_dim=3,  # u0, x, y
        output_dim=1  # u
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    num_epochs = 500
    train_losses = []
    val_losses = []
    
    print("Training FNO...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_input)
        loss = criterion(pred, train_output)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_input)
                val_loss = criterion(val_pred, val_output)
                val_losses.append(val_loss.item())
                
                print(f"Epoch {epoch:4d}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
        
        scheduler.step()
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_pred = model(test_input)
        test_loss = criterion(test_pred, test_output)
        print(f"Test Loss: {test_loss.item():.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    val_epochs = list(range(0, num_epochs, 10))
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize some results
    sample_idx = 0
    
    input_field = test_input[sample_idx, :, :, 0].cpu().numpy()
    true_output = test_output[sample_idx, :, :, 0].cpu().numpy()
    pred_output = test_pred[sample_idx, :, :, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(input_field, cmap='viridis')
    axes[0].set_title('Initial Condition')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(true_output, cmap='viridis')
    axes[1].set_title('True Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(pred_output, cmap='viridis')
    axes[2].set_title('FNO Prediction')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Compute relative error
    relative_error = np.linalg.norm(pred_output - true_output) / np.linalg.norm(true_output)
    print(f"Relative Error for sample {sample_idx}: {relative_error:.6f}")

if __name__ == "__main__":
    main()