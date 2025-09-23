"""
Example: Learning an operator using DeepONet

This example demonstrates how to use DeepONet to learn an operator
that maps an input function to its antiderivative.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ai4cfd.deeponet import DeepONet
from ai4cfd.utils import split_data, train_deeponet

def generate_antiderivative_data(num_samples=1000, num_sensors=100):
    """
    Generate training data for antiderivative operator.
    
    Input functions: Linear combinations of sine and cosine functions
    Output: Corresponding antiderivatives
    """
    # Sensor locations
    x_sensors = torch.linspace(0, 2*np.pi, num_sensors)
    
    # Evaluation points
    x_eval = torch.rand(num_samples) * 2 * np.pi
    
    # Generate random coefficients for basis functions
    num_modes = 5
    a_coeffs = torch.randn(num_samples, num_modes)
    b_coeffs = torch.randn(num_samples, num_modes)
    
    # Input functions evaluated at sensor locations
    branch_data = torch.zeros(num_samples, num_sensors)
    
    # Output (antiderivative) evaluated at random points
    trunk_data = x_eval.unsqueeze(1)  # Shape: (num_samples, 1)
    target_data = torch.zeros(num_samples, 1)
    
    for i in range(num_samples):
        # Input function: f(x) = sum(a_k * sin(k*x) + b_k * cos(k*x))
        f_vals = torch.zeros_like(x_sensors)
        antiderivative_val = 0.0
        
        for k in range(1, num_modes + 1):
            # Function values at sensors
            f_vals += a_coeffs[i, k-1] * torch.sin(k * x_sensors) + b_coeffs[i, k-1] * torch.cos(k * x_sensors)
            
            # Antiderivative value at evaluation point
            antiderivative_val += (-a_coeffs[i, k-1] / k) * torch.cos(k * x_eval[i]) + (b_coeffs[i, k-1] / k) * torch.sin(k * x_eval[i])
        
        branch_data[i] = f_vals
        target_data[i] = antiderivative_val
    
    return branch_data, trunk_data, target_data


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate training data
    print("Generating training data...")
    branch_data, trunk_data, target_data = generate_antiderivative_data(num_samples=5000, num_sensors=100)
    
    # Split data
    indices = torch.randperm(len(branch_data))
    train_size = int(0.8 * len(branch_data))
    val_size = int(0.1 * len(branch_data))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data loaders
    batch_size = 32
    
    train_dataset = torch.utils.data.TensorDataset(
        branch_data[train_indices], trunk_data[train_indices], target_data[train_indices]
    )
    val_dataset = torch.utils.data.TensorDataset(
        branch_data[val_indices], trunk_data[val_indices], target_data[val_indices]
    )
    test_dataset = torch.utils.data.TensorDataset(
        branch_data[test_indices], trunk_data[test_indices], target_data[test_indices]
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize DeepONet model
    model = DeepONet(
        branch_input_dim=100,  # Number of sensors
        trunk_input_dim=1,     # Evaluation coordinate (x)
        branch_hidden_dims=[128, 128, 128],
        trunk_hidden_dims=[128, 128, 128],
        output_dim=128,        # Number of basis functions
        activation="relu"
    )
    
    # Train the model
    print("Training DeepONet...")
    trained_model, loss_tracker = train_deeponet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        lr=1e-3,
        device=device,
        verbose=True
    )
    
    # Plot loss curves
    loss_tracker.plot()
    
    # Test the model
    trained_model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for branch_input, trunk_input, target in test_loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target = target.to(device)
            
            pred = trained_model(branch_input, trunk_input)
            loss = torch.mean((pred - target) ** 2)
            test_loss += loss.item()
            
            all_predictions.append(pred.cpu())
            all_targets.append(target.cpu())
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Visualize some results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(all_targets.numpy(), all_predictions.numpy(), alpha=0.6)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    
    plt.subplot(1, 2, 2)
    errors = (all_predictions - all_targets).numpy()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Compute relative error
    relative_error = torch.norm(all_predictions - all_targets) / torch.norm(all_targets)
    print(f"Relative Error: {relative_error:.6f}")

if __name__ == "__main__":
    main()