"""
测试不同初始条件的脚本
Test script for different initial conditions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_configuration, use_ic_preset

def plot_initial_condition(ic_type, preset_name=None):
    """
    绘制初始条件分布
    
    Args:
        ic_type: 初始条件类型
        preset_name: 预设名称 (可选)
    """
    # 获取配置
    if preset_name:
        use_ic_preset(preset_name)
    
    config = get_configuration()
    config['type'] = ic_type
    
    # 创建空间和速度网格
    x = torch.linspace(0, config['x_max'], 100)
    v = torch.linspace(-config['v_max'], config['v_max'], 100)
    X, V = torch.meshgrid(x, v, indexing='ij')
    
    # 创建临时类来计算初始条件
    class TempVPINN:
        def __init__(self, config):
            self.config = config
            
        def _initial_condition(self, x, v):
            """复制 VPINN 的初始条件方法"""
            ic_type = self.config.get('type', 'two_stream')
            
            if ic_type == 'two_stream':
                return self._ic_two_stream(x, v)
            elif ic_type == 'landau':
                return self._ic_landau(x, v)
            elif ic_type == 'single_beam':
                return self._ic_single_beam(x, v)
            elif ic_type == 'custom':
                custom_func = self.config.get('custom_ic_function')
                if custom_func is None:
                    raise ValueError("Custom IC function not provided!")
                return custom_func(x, v, self.config)
            else:
                raise ValueError(f"Unknown initial condition type: {ic_type}")
        
        def _ic_two_stream(self, x, v):
            k = 2 * torch.pi * self.config.get('perturb_mode', 1) / self.config['x_max']
            v_th = self.config.get('thermal_v', 0.5)
            v_b = self.config.get('beam_v', 1.0)
            alpha = self.config.get('perturb_amp', 0.1)
            
            norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
            term1 = norm_factor * torch.exp(-((v - v_b)**2) / (2 * v_th**2))
            term2 = norm_factor * torch.exp(-((v + v_b)**2) / (2 * v_th**2))
            
            return 0.5 * (term1 + term2) * (1 + alpha * torch.cos(k * x))
        
        def _ic_landau(self, x, v):
            k = 2 * torch.pi * self.config.get('landau_mode', 1) / self.config['x_max']
            v_th = self.config.get('landau_v_thermal', 1.0)
            alpha = self.config.get('landau_perturb_amp', 0.01)
            
            norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
            maxwell = norm_factor * torch.exp(-(v**2) / (2 * v_th**2))
            
            return maxwell * (1 + alpha * torch.cos(k * x))
        
        def _ic_single_beam(self, x, v):
            k = 2 * torch.pi * self.config.get('single_mode', 1) / self.config['x_max']
            v_c = self.config.get('single_v_center', 0.0)
            v_th = self.config.get('single_v_thermal', 0.5)
            alpha = self.config.get('single_perturb_amp', 0.05)
            
            norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
            maxwell = norm_factor * torch.exp(-((v - v_c)**2) / (2 * v_th**2))
            
            return maxwell * (1 + alpha * torch.cos(k * x))
    
    # 计算初始条件
    vpinn = TempVPINN(config)
    f0 = vpinn._initial_condition(X.flatten(), V.flatten())
    f0 = f0.reshape(X.shape).numpy()
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 相空间分布 f(x,v)
    im1 = axes[0].pcolormesh(X.numpy(), V.numpy(), f0, shading='auto', cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('v')
    axes[0].set_title(f'Phase Space: f(0,x,v)\n{ic_type}')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. 密度分布 n(x) = ∫f dv
    n_x = np.trapz(f0, v.numpy(), axis=1)
    axes[1].plot(x.numpy(), n_x, 'b-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('n(x)')
    axes[1].set_title('Density Profile')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 速度分布 f(v) at x=0
    x_idx = 0
    f_v = f0[x_idx, :]
    axes[2].plot(v.numpy(), f_v, 'r-', linewidth=2)
    axes[2].set_xlabel('v')
    axes[2].set_ylabel(f'f(v) at x={x.numpy()[x_idx]:.2f}')
    axes[2].set_title('Velocity Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_name = f'test_ic_{ic_type}'
    if preset_name:
        save_name = f'test_ic_{preset_name}'
    plt.savefig(f'{save_name}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}.png")
    
    plt.close()
    
    # 打印一些统计信息
    print(f"\n{'='*60}")
    print(f"Initial Condition: {ic_type}")
    if preset_name:
        print(f"Preset: {preset_name}")
    print(f"{'='*60}")
    print(f"Total mass: {np.trapz(n_x, x.numpy()):.6f}")
    print(f"Max density: {n_x.max():.6f}")
    print(f"Min density: {n_x.min():.6f}")
    print(f"Density variation: {(n_x.max() - n_x.min()) / n_x.mean() * 100:.2f}%")
    print(f"Max f(x,v): {f0.max():.6f}")
    print()


def test_all_presets():
    """测试所有预设初始条件"""
    print("\n" + "="*70)
    print("Testing All Initial Condition Presets")
    print("="*70 + "\n")
    
    presets = [
        ('two_stream', 'two_stream_strong'),
        ('two_stream', 'two_stream_weak'),
        ('landau', 'landau_damping'),
        ('single_beam', 'single_beam'),
    ]
    
    for ic_type, preset_name in presets:
        try:
            plot_initial_condition(ic_type, preset_name)
        except Exception as e:
            print(f"✗ Error testing {preset_name}: {e}")
    
    print("\n" + "="*70)
    print("All tests completed! Check the generated PNG files.")
    print("="*70 + "\n")


def compare_initial_conditions():
    """比较不同初始条件"""
    from config import get_configuration, use_ic_preset
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    presets = [
        ('two_stream_strong', 'Two-Stream (Strong)'),
        ('two_stream_weak', 'Two-Stream (Weak)'),
        ('landau_damping', 'Landau Damping'),
        ('single_beam', 'Single Beam'),
    ]
    
    for idx, (preset_name, title) in enumerate(presets):
        use_ic_preset(preset_name)
        config = get_configuration()
        
        # 创建网格
        x = torch.linspace(0, config['x_max'], 100)
        v = torch.linspace(-config['v_max'], config['v_max'], 100)
        X, V = torch.meshgrid(x, v, indexing='ij')
        
        # 使用临时 VPINN 计算
        class TempVPINN:
            def __init__(self, config):
                self.config = config
            
            def _initial_condition(self, x, v):
                ic_type = self.config.get('type', 'two_stream')
                if ic_type == 'two_stream':
                    return self._ic_two_stream(x, v)
                elif ic_type == 'landau':
                    return self._ic_landau(x, v)
                elif ic_type == 'single_beam':
                    return self._ic_single_beam(x, v)
            
            def _ic_two_stream(self, x, v):
                k = 2 * torch.pi * self.config.get('perturb_mode', 1) / self.config['x_max']
                v_th = self.config.get('thermal_v', 0.5)
                v_b = self.config.get('beam_v', 1.0)
                alpha = self.config.get('perturb_amp', 0.1)
                norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
                term1 = norm_factor * torch.exp(-((v - v_b)**2) / (2 * v_th**2))
                term2 = norm_factor * torch.exp(-((v + v_b)**2) / (2 * v_th**2))
                return 0.5 * (term1 + term2) * (1 + alpha * torch.cos(k * x))
            
            def _ic_landau(self, x, v):
                k = 2 * torch.pi * self.config.get('landau_mode', 1) / self.config['x_max']
                v_th = self.config.get('landau_v_thermal', 1.0)
                alpha = self.config.get('landau_perturb_amp', 0.01)
                norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
                maxwell = norm_factor * torch.exp(-(v**2) / (2 * v_th**2))
                return maxwell * (1 + alpha * torch.cos(k * x))
            
            def _ic_single_beam(self, x, v):
                k = 2 * torch.pi * self.config.get('single_mode', 1) / self.config['x_max']
                v_c = self.config.get('single_v_center', 0.0)
                v_th = self.config.get('single_v_thermal', 0.5)
                alpha = self.config.get('single_perturb_amp', 0.05)
                norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
                maxwell = norm_factor * torch.exp(-((v - v_c)**2) / (2 * v_th**2))
                return maxwell * (1 + alpha * torch.cos(k * x))
        
        vpinn = TempVPINN(config)
        f0 = vpinn._initial_condition(X.flatten(), V.flatten())
        f0 = f0.reshape(X.shape).numpy()
        
        # 绘制
        im = axes[idx].pcolormesh(X.numpy(), V.numpy(), f0, shading='auto', cmap='viridis')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('v')
        axes[idx].set_title(title)
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig('compare_initial_conditions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: compare_initial_conditions.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Initial Condition Testing Suite")
    print("="*70 + "\n")
    
    # 测试所有预设
    test_all_presets()
    
    # 生成对比图
    print("\nGenerating comparison plot...")
    compare_initial_conditions()
    
    print("\n✅ All tests completed successfully!")
    print("Check the generated PNG files for visualization.\n")
