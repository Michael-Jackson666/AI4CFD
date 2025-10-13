"""
对比实验脚本 - MLP vs Transformer
Compare MLP and Transformer architectures on the Vlasov-Poisson system.
"""

import torch
import numpy as np
from vp_pinn import VlasovPoissonPINN
import time


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
        'epochs': 1000,  # 减少到1000以便快速对比
        'learning_rate': 1e-4,
        'n_pde': 70000,
        'n_ic': 1100,
        'n_bc': 1100,

        # Loss Function Weights
        'weight_pde': 1.0,
        'weight_ic': 5.0,
        'weight_bc': 10.0,

        # Numerical & Logging Parameters
        'v_quad_points': 128,
        'log_frequency': 200,
        'plot_frequency': 500,
    }


def run_experiment(config_name, config, description):
    """运行单个实验"""
    print("\n" + "=" * 80)
    print(f"实验: {description}")
    print("=" * 80)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
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
        
        print(f"\n✓ 实验完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"  结果保存在: {config['plot_dir']}/")
        
        return {
            'success': True,
            'time': elapsed_time,
            'config_name': config_name
        }
        
    except Exception as e:
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
    print("  3. Standard Transformer (d=256, h=8, l=6)")
    print("\n每个模型训练 1000 epochs")
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
        'd_model': 256,
        'nhead': 8,
        'num_transformer_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'plot_dir': 'comparison/transformer_standard',
        'learning_rate': 5e-5,  # Transformer 通常需要更小的学习率
    })
    results.append(run_experiment('transformer_standard', config3, 'Standard Transformer'))
    
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
    
    model_choice = input("\n选择模型:\n  1. MLP\n  2. Lightweight Transformer\n  3. Standard Transformer\n\n输入选项 (1-3): ").strip()
    
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
            'plot_dir': 'quick_test/transformer_light'
        })
        description = 'Lightweight Transformer (快速测试)'
    elif model_choice == '3':
        config.update({
            'model_type': 'transformer',
            'd_model': 256,
            'nhead': 8,
            'num_transformer_layers': 6,
            'plot_dir': 'quick_test/transformer_standard',
            'learning_rate': 5e-5,
        })
        description = 'Standard Transformer (快速测试)'
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
