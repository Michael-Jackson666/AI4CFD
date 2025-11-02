"""
Main script for training the PINN on the Vlasov-Poisson system.
All configuration parameters are in config.py - modify that file to change settings.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from vp_pinn import VlasovPoissonPINN
from config import get_configuration, validate_configuration


def main():
    """
    Main function to run the PINN training.
    Configuration is loaded from config.py.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Disable Flash / efficient SDPA kernels to ensure higher-order autograd support
    if torch.backends.cuda.is_built():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    # ==================== Load Configuration ====================
    print("=" * 70)
    print("LOADING CONFIGURATION FROM config.py")
    print("=" * 70)
    
    try:
        # Validate configuration
        validate_configuration()
        
        # Get configuration dictionary
        configuration = get_configuration()
        
        print("✓ Configuration loaded and validated successfully!")
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        print("\nPlease check config.py and fix any errors.")
        return
    
    # ==================== Print Configuration ====================
    print("\n" + "=" * 70)
    print("VLASOV-POISSON PINN TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Domain: t ∈ [0, {configuration['t_max']}], "
          f"x ∈ [0, {configuration['x_max']}], "
          f"v ∈ [{-configuration['v_max']}, {configuration['v_max']}]")
    
    # Print model-specific info
    model_type = configuration.get('model_type', 'mlp')
    print(f"\n  Model Type: {model_type.upper()}")
    if model_type == 'mlp':
        print(f"  Network: {configuration['nn_layers']} layers × {configuration['nn_neurons']} neurons")
    elif 'transformer' in model_type:
        print(f"  d_model: {configuration.get('d_model', 256)}, "
              f"heads: {configuration.get('nhead', 8)}, "
              f"layers: {configuration.get('num_transformer_layers', 6)}")
        if model_type == 'hybrid_transformer':
            print(f"  MLP branch: {configuration.get('num_mlp_layers', 4)} layers × "
                  f"{configuration.get('mlp_neurons', 512)} neurons")
    
    print(f"\n  Training: {configuration['epochs']} epochs, LR={configuration['learning_rate']}")
    print(f"  Sampling: PDE={configuration['n_pde']}, IC={configuration['n_ic']}, BC={configuration['n_bc']}")
    print(f"  Loss weights: PDE={configuration['weight_pde']}, "
          f"IC={configuration['weight_ic']}, BC={configuration['weight_bc']}")
    print("=" * 70)
    
    # ==================== Save Configuration ====================
    # 创建输出目录
    os.makedirs(configuration['plot_dir'], exist_ok=True)
    
    # 添加训练开始时间和系统信息
    config_to_save = configuration.copy()
    config_to_save['training_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_to_save['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_to_save['torch_version'] = torch.__version__
    config_to_save['numpy_version'] = np.__version__
    
    # 保存为 JSON 格式
    config_json_path = os.path.join(configuration['plot_dir'], 'training_config.json')
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)
    
    # 保存为易读的文本格式
    config_txt_path = os.path.join(configuration['plot_dir'], 'training_config.txt')
    with open(config_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("VLASOV-POISSON PINN TRAINING CONFIGURATION\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nTraining Start Time: {config_to_save['training_start_time']}\n")
        f.write(f"Device: {config_to_save['device']}\n")
        f.write(f"PyTorch Version: {config_to_save['torch_version']}\n")
        f.write(f"NumPy Version: {config_to_save['numpy_version']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("DOMAIN PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"t_max: {configuration['t_max']}\n")
        f.write(f"x_max: {configuration['x_max']}\n")
        f.write(f"v_max: {configuration['v_max']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("PHYSICS PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"beam_v: {configuration['beam_v']}\n")
        f.write(f"thermal_v: {configuration['thermal_v']}\n")
        f.write(f"perturb_amp: {configuration['perturb_amp']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("=" * 70 + "\n")
        f.write(f"model_type: {configuration['model_type']}\n")
        if configuration['model_type'] == 'mlp':
            f.write(f"nn_layers: {configuration['nn_layers']}\n")
            f.write(f"nn_neurons: {configuration['nn_neurons']}\n")
        elif 'transformer' in configuration['model_type']:
            f.write(f"d_model: {configuration.get('d_model', 256)}\n")
            f.write(f"nhead: {configuration.get('nhead', 8)}\n")
            f.write(f"num_transformer_layers: {configuration.get('num_transformer_layers', 6)}\n")
            f.write(f"dim_feedforward: {configuration.get('dim_feedforward', 1024)}\n")
            f.write(f"dropout: {configuration.get('dropout', 0.1)}\n")
            if configuration['model_type'] == 'hybrid_transformer':
                f.write(f"num_mlp_layers: {configuration.get('num_mlp_layers', 4)}\n")
                f.write(f"mlp_neurons: {configuration.get('mlp_neurons', 512)}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAINING HYPERPARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"epochs: {configuration['epochs']}\n")
        f.write(f"learning_rate: {configuration['learning_rate']}\n")
        f.write(f"n_pde: {configuration['n_pde']}\n")
        f.write(f"n_ic: {configuration['n_ic']}\n")
        f.write(f"n_bc: {configuration['n_bc']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("LOSS FUNCTION WEIGHTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"weight_pde: {configuration['weight_pde']}\n")
        f.write(f"weight_ic: {configuration['weight_ic']}\n")
        f.write(f"weight_bc: {configuration['weight_bc']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("NUMERICAL & LOGGING PARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"v_quad_points: {configuration['v_quad_points']}\n")
        f.write(f"log_frequency: {configuration['log_frequency']}\n")
        f.write(f"plot_frequency: {configuration['plot_frequency']}\n")
        f.write(f"plot_dir: {configuration['plot_dir']}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Configuration saved to:")
    print(f"  - {config_json_path}")
    print(f"  - {config_txt_path}")
    
    # ==================== Initialize and Train ====================
    try:
        # Create PINN solver
        pinn_solver = VlasovPoissonPINN(configuration)
        
        # Run training
        pinn_solver.train()
        
        # 保存训练完成信息
        training_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 更新配置文件，添加训练结束时间
        config_to_save['training_end_time'] = training_end_time
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        
        # 追加训练完成信息到文本文件
        with open(config_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\nTraining End Time: {training_end_time}\n")
            f.write("=" * 70 + "\n")
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print(f"Results saved to: {configuration['plot_dir']}/")
        print(f"Configuration files:")
        print(f"  - training_config.json")
        print(f"  - training_config.txt")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be available in the output directory.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()