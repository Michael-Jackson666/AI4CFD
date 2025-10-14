"""
数据生成模块 - 为 DeepONet 生成 Vlasov-Poisson 训练数据
Data Generation Module for DeepONet Training on Vlasov-Poisson System

生成策略：
1. 使用解析解或数值求解器生成初始条件-演化结果对
2. 变化初始条件参数（束流速度、热速度、扰动幅度等）
3. 采样不同的时空点作为训练数据
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle


class VlasovPoissonDataGenerator:
    """
    Vlasov-Poisson 数据生成器
    
    使用特征线方法或有限差分方法求解 Vlasov-Poisson 系统，
    生成不同初始条件下的演化数据。
    """
    
    def __init__(self, config):
        """
        初始化数据生成器
        
        Args:
            config (dict): 配置参数
        """
        self.config = config
        
        # 物理域设置
        self.t_max = config.get('t_max', 50.0)
        self.x_max = config.get('x_max', 10.0)
        self.v_max = config.get('v_max', 5.0)
        
        # 网格设置
        self.nx = config.get('nx', 64)
        self.nv = config.get('nv', 64)
        self.nt = config.get('nt', 100)
        
        # 创建网格
        self.x = np.linspace(0, self.x_max, self.nx)
        self.v = np.linspace(-self.v_max, self.v_max, self.nv)
        self.t = np.linspace(0, self.t_max, self.nt)
        
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]
        self.dt = self.t[1] - self.t[0]
        
        print(f"数据生成器初始化:")
        print(f"  空间网格: {self.nx} 点, dx={self.dx:.4f}")
        print(f"  速度网格: {self.nv} 点, dv={self.dv:.4f}")
        print(f"  时间步数: {self.nt} 步, dt={self.dt:.4f}")
    
    def initial_condition(self, x, v, beam_v=1.0, thermal_v=0.5, perturb_amp=0.1, k_mode=1):
        """
        生成双流不稳定性初始条件
        
        Args:
            x: 空间坐标数组
            v: 速度坐标数组
            beam_v: 束流速度
            thermal_v: 热速度
            perturb_amp: 扰动幅度
            k_mode: 波数模式
            
        Returns:
            f0: 初始分布函数 [nx, nv]
        """
        X, V = np.meshgrid(x, v, indexing='ij')
        
        # 归一化因子
        norm = 1.0 / (thermal_v * np.sqrt(2 * np.pi))
        
        # 双 Maxwellian
        f1 = norm * np.exp(-(V - beam_v)**2 / (2 * thermal_v**2))
        f2 = norm * np.exp(-(V + beam_v)**2 / (2 * thermal_v**2))
        
        # 加入空间扰动
        k = 2 * np.pi * k_mode / self.x_max
        perturbation = 1 + perturb_amp * np.cos(k * X)
        
        f0 = 0.5 * (f1 + f2) * perturbation
        
        return f0
    
    def compute_electric_field(self, f):
        """
        通过 Poisson 方程计算电场
        
        Args:
            f: 分布函数 [nx, nv]
            
        Returns:
            E: 电场 [nx]
        """
        # 计算电子密度: n_e = ∫ f dv
        n_e = np.trapz(f, self.v, axis=1)
        
        # 电荷密度偏差: ρ = n_e - 1
        rho = n_e - 1.0
        
        # 通过积分求解电场: dE/dx = ρ
        E = np.zeros_like(n_e)
        E[0] = 0  # 边界条件
        for i in range(1, len(E)):
            E[i] = E[i-1] + rho[i-1] * self.dx
        
        # 周期性边界：去除平均值
        E = E - np.mean(E)
        
        return E
    
    def solve_vlasov_splitting(self, f0, n_steps=None):
        """
        使用算子分裂法求解 Vlasov-Poisson 系统
        
        分裂方案:
        1. 对流步: ∂f/∂t + v·∂f/∂x = 0
        2. 加速步: ∂f/∂t - E·∂f/∂v = 0
        
        Args:
            f0: 初始条件 [nx, nv]
            n_steps: 时间步数（如果为None则使用self.nt）
            
        Returns:
            f_history: 演化历史 [nt, nx, nv]
            E_history: 电场历史 [nt, nx]
        """
        if n_steps is None:
            n_steps = self.nt
        
        f_history = np.zeros((n_steps, self.nx, self.nv))
        E_history = np.zeros((n_steps, self.nx))
        
        f = f0.copy()
        f_history[0] = f
        E_history[0] = self.compute_electric_field(f)
        
        print("使用算子分裂法求解 Vlasov-Poisson...")
        
        for n in tqdm(range(1, n_steps), desc="时间演化"):
            # 计算电场
            E = self.compute_electric_field(f)
            
            # Step 1: 对流步 (∂f/∂t + v·∂f/∂x = 0)
            f_temp = np.zeros_like(f)
            for j in range(self.nv):
                v_j = self.v[j]
                # 上风格式
                if v_j > 0:
                    for i in range(self.nx):
                        i_minus = (i - 1) % self.nx
                        f_temp[i, j] = f[i, j] - v_j * self.dt / self.dx * (f[i, j] - f[i_minus, j])
                else:
                    for i in range(self.nx):
                        i_plus = (i + 1) % self.nx
                        f_temp[i, j] = f[i, j] - v_j * self.dt / self.dx * (f[i_plus, j] - f[i, j])
            
            # Step 2: 加速步 (∂f/∂t - E·∂f/∂v = 0)
            f_new = np.zeros_like(f)
            for i in range(self.nx):
                E_i = E[i]
                # 上风格式
                if E_i > 0:
                    for j in range(self.nv):
                        if j > 0:
                            f_new[i, j] = f_temp[i, j] + E_i * self.dt / self.dv * (f_temp[i, j] - f_temp[i, j-1])
                        else:
                            f_new[i, j] = f_temp[i, j]
                else:
                    for j in range(self.nv):
                        if j < self.nv - 1:
                            f_new[i, j] = f_temp[i, j] + E_i * self.dt / self.dv * (f_temp[i, j+1] - f_temp[i, j])
                        else:
                            f_new[i, j] = f_temp[i, j]
            
            # 确保非负
            f_new = np.maximum(f_new, 0)
            
            f = f_new
            f_history[n] = f
            E_history[n] = E
        
        return f_history, E_history
    
    def generate_dataset(self, n_samples=100, output_dir='data'):
        """
        生成训练数据集
        
        变化参数:
        - beam_v: 束流速度 [0.5, 2.0]
        - thermal_v: 热速度 [0.02, 0.5]
        - perturb_amp: 扰动幅度 [0.05, 0.2]
        - k_mode: 波数模式 [1, 3]
        
        Args:
            n_samples: 样本数量
            output_dir: 输出目录
            
        Returns:
            dataset: 包含训练数据的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n生成 {n_samples} 个训练样本...")
        
        # 参数范围
        beam_v_range = (0.5, 2.0)
        thermal_v_range = (0.02, 0.5)
        perturb_amp_range = (0.05, 0.2)
        k_mode_range = (1, 3)
        
        dataset = {
            'initial_conditions': [],  # 初始条件 [n_samples, nx, nv]
            'solutions': [],           # 完整解 [n_samples, nt, nx, nv]
            'electric_fields': [],     # 电场 [n_samples, nt, nx]
            'parameters': [],          # 参数 [n_samples, 4]
            'x': self.x,
            'v': self.v,
            't': self.t
        }
        
        for i in tqdm(range(n_samples), desc="生成样本"):
            # 随机采样参数
            beam_v = np.random.uniform(*beam_v_range)
            thermal_v = np.random.uniform(*thermal_v_range)
            perturb_amp = np.random.uniform(*perturb_amp_range)
            k_mode = np.random.randint(*k_mode_range)
            
            # 生成初始条件
            f0 = self.initial_condition(
                self.x, self.v,
                beam_v=beam_v,
                thermal_v=thermal_v,
                perturb_amp=perturb_amp,
                k_mode=k_mode
            )
            
            # 求解演化
            f_history, E_history = self.solve_vlasov_splitting(f0)
            
            # 保存数据
            dataset['initial_conditions'].append(f0)
            dataset['solutions'].append(f_history)
            dataset['electric_fields'].append(E_history)
            dataset['parameters'].append([beam_v, thermal_v, perturb_amp, k_mode])
        
        # 转换为 numpy 数组
        dataset['initial_conditions'] = np.array(dataset['initial_conditions'])
        dataset['solutions'] = np.array(dataset['solutions'])
        dataset['electric_fields'] = np.array(dataset['electric_fields'])
        dataset['parameters'] = np.array(dataset['parameters'])
        
        # 保存数据集
        save_path = os.path.join(output_dir, 'vp_dataset.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n数据集已保存到: {save_path}")
        print(f"数据形状:")
        print(f"  初始条件: {dataset['initial_conditions'].shape}")
        print(f"  解: {dataset['solutions'].shape}")
        print(f"  电场: {dataset['electric_fields'].shape}")
        print(f"  参数: {dataset['parameters'].shape}")
        
        # 保存可视化样本
        self.visualize_samples(dataset, output_dir, n_vis=5)
        
        return dataset
    
    def visualize_samples(self, dataset, output_dir, n_vis=5):
        """
        可视化数据样本
        
        Args:
            dataset: 数据集字典
            output_dir: 输出目录
            n_vis: 可视化样本数
        """
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\n生成 {n_vis} 个样本可视化...")
        
        for i in range(min(n_vis, len(dataset['initial_conditions']))):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            params = dataset['parameters'][i]
            f0 = dataset['initial_conditions'][i]
            f_sol = dataset['solutions'][i]
            E_sol = dataset['electric_fields'][i]
            
            # 初始条件
            im0 = axes[0, 0].contourf(self.x, self.v, f0.T, levels=20, cmap='viridis')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('v')
            axes[0, 0].set_title(f'Initial Condition\nbeam_v={params[0]:.2f}, thermal_v={params[1]:.3f}')
            plt.colorbar(im0, ax=axes[0, 0])
            
            # 不同时刻的相空间
            time_indices = [len(self.t)//4, len(self.t)//2, -1]
            titles = ['t = T/4', 't = T/2', 't = T']
            
            for idx, (t_idx, title) in enumerate(zip(time_indices, titles)):
                ax = axes[0, idx+1] if idx < 2 else axes[1, 0]
                im = ax.contourf(self.x, self.v, f_sol[t_idx].T, levels=20, cmap='viridis')
                ax.set_xlabel('x')
                ax.set_ylabel('v')
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
            
            # 电场演化
            T, X = np.meshgrid(self.t, self.x, indexing='ij')
            im_E = axes[1, 1].contourf(X, T, E_sol, levels=20, cmap='RdBu_r')
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('t')
            axes[1, 1].set_title('Electric Field Evolution')
            plt.colorbar(im_E, ax=axes[1, 1])
            
            # 密度演化
            n_e_history = np.array([np.trapz(f_sol[t], self.v, axis=1) for t in range(len(self.t))])
            im_n = axes[1, 2].contourf(X, T, n_e_history, levels=20, cmap='plasma')
            axes[1, 2].set_xlabel('x')
            axes[1, 2].set_ylabel('t')
            axes[1, 2].set_title('Density Evolution')
            plt.colorbar(im_n, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'sample_{i:03d}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"可视化已保存到: {vis_dir}")


def main():
    """
    主函数：生成训练数据集
    """
    # 配置参数
    config = {
        # 物理域
        't_max': 50.0,
        'x_max': 10.0,
        'v_max': 5.0,
        
        # 网格分辨率
        'nx': 64,      # 空间网格点数
        'nv': 64,      # 速度网格点数
        'nt': 100,     # 时间步数
    }
    
    # 创建数据生成器
    generator = VlasovPoissonDataGenerator(config)
    
    # 生成数据集
    # 训练集
    print("\n" + "="*70)
    print("生成训练集")
    print("="*70)
    dataset_train = generator.generate_dataset(
        n_samples=100,
        output_dir='data/train'
    )
    
    # 验证集
    print("\n" + "="*70)
    print("生成验证集")
    print("="*70)
    dataset_val = generator.generate_dataset(
        n_samples=20,
        output_dir='data/val'
    )
    
    # 测试集
    print("\n" + "="*70)
    print("生成测试集")
    print("="*70)
    dataset_test = generator.generate_dataset(
        n_samples=20,
        output_dir='data/test'
    )
    
    print("\n" + "="*70)
    print("数据生成完成！")
    print("="*70)
    print("\n数据集统计:")
    print(f"  训练集: {len(dataset_train['initial_conditions'])} 样本")
    print(f"  验证集: {len(dataset_val['initial_conditions'])} 样本")
    print(f"  测试集: {len(dataset_test['initial_conditions'])} 样本")
    print(f"\n数据保存在: data/ 目录")


if __name__ == '__main__':
    main()
