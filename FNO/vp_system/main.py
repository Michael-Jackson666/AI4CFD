"""
Main training script for FNO Vlasov-Poisson System.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from config import Config, get_default_config, get_fast_test_config
from vp_fno import create_vp_fno_model
from transformer import create_vp_transformer_model
from data_generator import create_dataloaders, VPDataGenerator
from visualization import VPVisualizer, visualize_model_predictions


class Trainer:
    """
    Trainer class for VP-FNO models.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Set random seeds
        config.set_seed()
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # TensorBoard writer
        if config.training.use_tensorboard:
            log_dir = os.path.join(config.training.tensorboard_dir, 
                                  config.experiment_name,
                                  datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_l2_error': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def _create_model(self):
        """Create model based on configuration."""
        if self.config.model_name == "fno":
            model = create_vp_fno_model(self.config.fno, self.config.fno.model_type)
        elif self.config.model_name == "transformer":
            model = create_vp_transformer_model(self.config.transformer, "standard")
        elif self.config.model_name == "hybrid":
            model = create_vp_transformer_model(self.config.transformer, "hybrid")
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        # Print model info
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {self.config.model_name}")
        print(f"Parameters: {n_params:,}")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.training.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_rate
            )
        elif self.config.training.scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_decay_rate
            )
        elif self.config.training.scheduler_type == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.scheduler_type}")
        
        return scheduler
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.training.epochs}')
        
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            data_loss = self.criterion(outputs, targets)
            loss = data_loss * self.config.training.data_loss_weight
            
            # Physics-informed loss (if enabled)
            if self.config.training.use_physics_loss and hasattr(self.model, 'compute_electric_field'):
                E = self.model.compute_electric_field(
                    outputs, 
                    dx=self.config.physics.dx,
                    dv=self.config.physics.dv
                )
                # Add physics residual loss (placeholder - implement actual residual)
                physics_loss = torch.tensor(0.0).to(self.device)
                loss += physics_loss * self.config.training.physics_loss_weight
            
            # Conservation loss (if enabled)
            if self.config.training.use_conservation_loss and hasattr(self.model, 'compute_conservation_loss'):
                f0 = inputs[:, 0:1, :, :]  # Initial distribution
                mass_loss, energy_loss = self.model.compute_conservation_loss(
                    outputs, f0, dv=self.config.physics.dv
                )
                conservation_loss = (mass_loss * self.config.training.mass_conservation_weight +
                                    energy_loss * self.config.training.energy_conservation_weight)
                loss += conservation_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.grad_clip
                )
            
            self.optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_l2_error = 0.0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Compute L2 error
                l2_error = torch.norm(outputs - targets) / torch.norm(targets)
                total_l2_error += l2_error.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_l2_error = total_l2_error / len(val_loader)
        
        return avg_loss, avg_l2_error
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config.training.save_every == 0:
            checkpoint_path = os.path.join(
                self.config.training.checkpoint_dir,
                f'{self.config.experiment_name}_epoch_{epoch+1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.training.checkpoint_dir,
                f'{self.config.experiment_name}_best.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch+1}")
        return epoch
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        for epoch in range(self.config.training.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_l2_error = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_l2_error'].append(val_l2_error)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Logging
            if (epoch + 1) % self.config.training.log_interval == 0:
                print(f"\nEpoch {epoch+1}/{self.config.training.epochs}:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Val L2 Error: {val_l2_error:.6f}")
                print(f"  Learning Rate: {current_lr:.6e}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Error/val_l2', val_l2_error, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.training.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        print("\n" + "="*80)
        print("Training Completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*80)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.history


def main(args):
    """Main function."""
    # Load configuration
    if args.config == "default":
        config = get_default_config()
    elif args.config == "fast":
        config = get_fast_test_config()
    else:
        # Load from file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    
    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Print configuration
    config.print_config()
    
    # Create data loaders
    print("\nPreparing data...")
    dataset_path = args.dataset if args.dataset else None
    train_loader, val_loader, test_loader = create_dataloaders(
        config.training,
        config.physics,
        dataset_path=dataset_path
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    if not args.test_only:
        history = trainer.train(train_loader, val_loader)
        
        # Save training history
        history_path = os.path.join(
            config.training.checkpoint_dir,
            f'{config.experiment_name}_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
        
        # Plot training history
        from visualization import VPVisualizer
        x, v, _ = config.physics.get_grids()
        visualizer = VPVisualizer(x, v)
        history_plot_path = os.path.join(
            config.training.checkpoint_dir,
            f'{config.experiment_name}_history.png'
        )
        visualizer.plot_training_history(history, save_path=history_plot_path)
    
    # Test model
    print("\nTesting model...")
    test_loss, test_l2_error = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test L2 Error: {test_l2_error:.6f}")
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating visualizations...")
        x, v, _ = config.physics.get_grids()
        results_dir = os.path.join(config.training.checkpoint_dir, 'visualizations')
        visualize_model_predictions(
            trainer.model,
            test_loader,
            trainer.device,
            x, v,
            n_samples=args.n_vis_samples,
            save_dir=results_dir
        )
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FNO for Vlasov-Poisson System')
    
    # Configuration
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration: "default", "fast", or path to config file')
    parser.add_argument('--model', type=str, choices=['fno', 'transformer', 'hybrid'],
                       help='Model type (overrides config)')
    parser.add_argument('--experiment-name', type=str,
                       help='Experiment name (overrides config)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Data
    parser.add_argument('--dataset', type=str,
                       help='Path to existing dataset (HDF5 file)')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume training')
    
    # Testing
    parser.add_argument('--test-only', action='store_true',
                       help='Only test, do not train')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations after training')
    parser.add_argument('--n-vis-samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    main(args)
