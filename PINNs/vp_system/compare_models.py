"""
对比实验脚本 - MLP vs Transformer
Compare MLP and Transformer architectures on the Vlasov-Poisson system.
"""

import os
import json
import time
from datetime import datetime

import torch
import numpy as np
from vp_pinn import VlasovPoissonPINN


def get_base_config():
    """基础配置"""
    return {
        # Domain Parameters
        't_max': 62.5,
        'x_max': 10.0,
        'v_max': 5.0,

        # Physics Parameters
        'beam_v': 1.0,
        'thermal_v': 0.5,
        'perturb_amp': 0.1,

        # Training Hyperparameters
        'epochs': 2000,  
        'learning_rate': 1e-4,
        'n_pde': 16000,
        'n_ic': 900,
        'n_bc': 900,

        # Loss Function Weights
        'weight_pde': 1.0,
        'weight_ic': 5.0,
        'weight_bc': 10.0,

        # Numerical & Logging Parameters
        'v_quad_points': 64,
        'log_frequency': 200,
        'plot_frequency': 500,
    }


def _save_initial_config(config, config_name, description):
    """保存初始配置，返回可更新的字典及文件路径"""
    plot_dir = config.get('plot_dir') or os.path.join('comparison', config_name)
    config['plot_dir'] = plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    config_to_save = dict(config)
    config_to_save['config_name'] = config_name
    config_to_save['experiment_description'] = description
    config_to_save['training_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_to_save['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_to_save['torch_version'] = torch.__version__
    config_to_save['numpy_version'] = np.__version__

    config_json_path = os.path.join(plot_dir, 'training_config.json')
    config_txt_path = os.path.join(plot_dir, 'training_config.txt')

    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)

    header_line = "=" * 70 + "\n"
    with open(config_txt_path, 'w', encoding='utf-8') as f:
        f.write(header_line)
        f.write("VLASOV-POISSON PINN TRAINING CONFIGURATION\n")
        f.write(header_line)
        f.write(f"\nExperiment: {description}\n")
        f.write(f"Config Name: {config_name}\n")
        f.write(f"Training Start Time: {config_to_save['training_start_time']}\n")
        f.write(f"Device: {config_to_save['device']}\n")
        f.write(f"PyTorch Version: {config_to_save['torch_version']}\n")
        f.write(f"NumPy Version: {config_to_save['numpy_version']}\n")
        f.write("\n" + header_line)
        f.write("DOMAIN PARAMETERS\n")
        f.write(header_line)
        f.write(f"t_max: {config['t_max']}\n")
        f.write(f"x_max: {config['x_max']}\n")
        f.write(f"v_max: {config['v_max']}\n")
        f.write("\n" + header_line)
        f.write("PHYSICS PARAMETERS\n")
        f.write(header_line)
        f.write(f"beam_v: {config['beam_v']}\n")
        f.write(f"thermal_v: {config['thermal_v']}\n")
        f.write(f"perturb_amp: {config['perturb_amp']}\n")
        f.write("\n" + header_line)
        f.write("MODEL ARCHITECTURE\n")
        f.write(header_line)
        f.write(f"model_type: {config['model_type']}\n")
        if config['model_type'] == 'mlp':
            f.write(f"nn_layers: {config['nn_layers']}\n")
            f.write(f"nn_neurons: {config['nn_neurons']}\n")
        elif 'transformer' in config['model_type']:
            f.write(f"d_model: {config.get('d_model', 'N/A')}\n")
            f.write(f"nhead: {config.get('nhead', 'N/A')}\n")
            f.write(f"num_transformer_layers: {config.get('num_transformer_layers', 'N/A')}\n")
            f.write(f"dim_feedforward: {config.get('dim_feedforward', 'N/A')}\n")
            f.write(f"dropout: {config.get('dropout', 'N/A')}\n")
        f.write("\n" + header_line)
        f.write("TRAINING HYPERPARAMETERS\n")
        f.write(header_line)
        f.write(f"epochs: {config['epochs']}\n")
        f.write(f"learning_rate: {config['learning_rate']}\n")
        f.write(f"n_pde: {config['n_pde']}\n")
        f.write(f"n_ic: {config['n_ic']}\n")
        f.write(f"n_bc: {config['n_bc']}\n")
        f.write("\n" + header_line)
        f.write("LOSS FUNCTION WEIGHTS\n")
        f.write(header_line)
        f.write(f"weight_pde: {config['weight_pde']}\n")
        f.write(f"weight_ic: {config['weight_ic']}\n")
        f.write(f"weight_bc: {config['weight_bc']}\n")
        f.write("\n" + header_line)
        f.write("NUMERICAL & LOGGING PARAMETERS\n")
        f.write(header_line)
        f.write(f"v_quad_points: {config['v_quad_points']}\n")
        f.write(f"log_frequency: {config['log_frequency']}\n")
        f.write(f"plot_frequency: {config['plot_frequency']}\n")
        f.write(f"plot_dir: {config['plot_dir']}\n")
        f.write(header_line)

    print(f"配置文件已保存至: {config_json_path}")
    return config_to_save, config_json_path, config_txt_path


def run_experiment(config_name, config, description):
    """运行单个实验"""
    print("\n" + "=" * 80)
    print(f"实验: {description}")
    print("=" * 80)
    
    # Create a working copy to avoid mutating caller state
    config = dict(config)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # 保障二阶梯度时注意力的反向传播可用
    if torch.backends.cuda.is_built():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    config_to_save, config_json_path, config_txt_path = _save_initial_config(config, config_name, description)

    # Print configuration
    model_type = config['model_type']
    print(f"模型类型: {model_type}")
    
    if model_type == 'mlp':
        print(f"  网络: {config['nn_layers']} 层 × {config['nn_neurons']} 神经元")
    elif 'transformer' in model_type:
        print(f"  d_model: {config.get('d_model', 256)}")
        print(f"  注意力头: {config.get('nhead', 8)}")
        print(f"  Transformer层: {config.get('num_transformer_layers', 6)}")
    
    print(f"输出目录: {config['plot_dir']}")
    
    try:
        start_time = time.time()
        
        # Create and train
        pinn_solver = VlasovPoissonPINN(config)
        pinn_solver.train()
        
        elapsed_time = time.time() - start_time

        training_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config_to_save['training_end_time'] = training_end_time
        config_to_save['status'] = 'success'
        config_to_save['training_duration_seconds'] = elapsed_time
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        with open(config_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\nTraining End Time: {training_end_time}\n")
            f.write(f"Status: success\n")
            f.write(f"Training Duration (s): {elapsed_time:.3f}\n")
            f.write("=" * 70 + "\n")
        
        print(f"\n✓ 实验完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"  结果保存在: {config['plot_dir']}/")
        
        return {
            'success': True,
            'time': elapsed_time,
            'config_name': config_name
        }
        
    except Exception as e:
        training_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config_to_save['training_end_time'] = training_end_time
        config_to_save['status'] = 'failed'
        config_to_save['error_message'] = str(e)
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        with open(config_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\nTraining End Time: {training_end_time}\n")
            f.write("Status: failed\n")
            f.write(f"Error: {e}\n")
            f.write("=" * 70 + "\n")
        print(f"\n✗ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'time': 0,
            'config_name': config_name
        }


def compare_mlp_vs_transformer():
    """对比 MLP 和 Transformer"""
    print("\n" + "=" * 80)
    print("MLP vs TRANSFORMER 对比实验")
    print("=" * 80)
    print("\n将训练以下模型:")
    print("  1. MLP - Standard (8层×128神经元)")
    print("  2. Lightweight Transformer (d=128, h=4, l=3)")
    print("  3. Compact Transformer (d=192, h=6, l=4)")
    print("\n每个模型训练 800 epochs，采样数量已下调以提升稳定性")
    print("=" * 80)
    
    input("\n按回车键开始实验...")
    
    results = []
    
    # 实验 1: MLP Standard
    config1 = get_base_config()
    config1.update({
        'model_type': 'mlp',
        'nn_layers': 8,
        'nn_neurons': 128,
        'plot_dir': 'comparison/mlp_standard'
    })
    results.append(run_experiment('mlp_standard', config1, 'MLP Standard (8×128)'))
    
    # 实验 2: Lightweight Transformer
    config2 = get_base_config()
    config2.update({
        'model_type': 'lightweight_transformer',
        'd_model': 128,
        'nhead': 4,
        'num_transformer_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'plot_dir': 'comparison/transformer_light'
    })
    results.append(run_experiment('transformer_light', config2, 'Lightweight Transformer'))
    
    # 实验 3: Standard Transformer
    config3 = get_base_config()
    config3.update({
        'model_type': 'transformer',
        'd_model': 192,
        'nhead': 6,
        'num_transformer_layers': 4,
        'dim_feedforward': 768,
        'dropout': 0.1,
        'plot_dir': 'comparison/transformer_standard',
        'learning_rate': 8e-5,  # Transformer 通常需要更小的学习率
    })
    results.append(run_experiment('transformer_standard', config3, 'Compact Transformer'))
    
    # 打印总结
    print("\n\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    
    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
        time_str = f"{result['time']/60:.2f} 分钟" if result['success'] else "N/A"
        print(f"  [{status}] {result['config_name']:<20} - 训练时间: {time_str}")
    
    print("\n" + "=" * 80)
    print("查看结果:")
    print("  comparison/mlp_standard/")
    print("  comparison/transformer_light/")
    print("  comparison/transformer_standard/")
    print("=" * 80)


def quick_test():
    """快速测试 - 少量epochs"""
    print("\n" + "=" * 80)
    print("快速测试模式 (200 epochs)")
    print("=" * 80)
    
    model_choice = input("\n选择模型:\n  1. MLP\n  2. Lightweight Transformer\n  3. Compact Transformer\n\n输入选项 (1-3): ").strip()
    
    config = get_base_config()
    config['epochs'] = 200
    config['log_frequency'] = 50
    config['plot_frequency'] = 100
    
    if model_choice == '1':
        config.update({
            'model_type': 'mlp',
            'nn_layers': 8,
            'nn_neurons': 128,
            'plot_dir': 'quick_test/mlp'
        })
        description = 'MLP (快速测试)'
    elif model_choice == '2':
        config.update({
            'model_type': 'lightweight_transformer',
            'd_model': 128,
            'nhead': 4,
            'num_transformer_layers': 3,
            'dim_feedforward': 512,
            'plot_dir': 'quick_test/transformer_light'
        })
        description = 'Lightweight Transformer (快速测试)'
    elif model_choice == '3':
        config.update({
            'model_type': 'transformer',
            'd_model': 192,
            'nhead': 6,
            'num_transformer_layers': 4,
            'dim_feedforward': 768,
            'plot_dir': 'quick_test/transformer_standard',
            'learning_rate': 8e-5,
        })
        description = 'Compact Transformer (快速测试)'
    else:
        print("无效选择")
        return
    
    run_experiment('quick_test', config, description)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("MLP vs TRANSFORMER 对比实验工具")
    print("=" * 80)
    print("\n选项:")
    print("  1. 完整对比实验 (MLP vs Lightweight vs Standard Transformer)")
    print("  2. 快速测试 (选择一个模型，200 epochs)")
    print("  3. 退出")
    
    choice = input("\n输入选项 (1-3): ").strip()
    
    if choice == '1':
        compare_mlp_vs_transformer()
    elif choice == '2':
        quick_test()
    elif choice == '3':
        print("退出")
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
