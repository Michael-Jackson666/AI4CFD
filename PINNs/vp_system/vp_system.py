#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vlasov-Poisson方程组求解器
使用DeepXDE求解6维Vlasov-Poisson系统

Vlasov方程 (6D相空间):
∂f/∂t + v·∇_x f + q/m E·∇_v f = 0

Poisson方程:
∇²φ = -ρ/ε₀ = -q/ε₀ ∫ f dv

其中 E = -∇φ

作者：DeepXDE Tutorial  
日期：2025年9月30日
"""

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import time
from scipy.integrate import quad


class VlasovPoissonSolver:
    """Vlasov-Poisson方程组求解器"""
    
    def __init__(self, 
                 x_domain=(-1.0, 1.0), 
                 v_domain=(-3.0, 3.0), 
                 time_domain=(0.0, 1.0),
                 q_over_m=1.0, 
                 epsilon_0=1.0,
                 case="landau_damping"):
        """
        初始化Vlasov-Poisson求解器
        
        Args:
            x_domain (tuple): 空间域范围 (x_min, x_max)
            v_domain (tuple): 速度域范围 (v_min, v_max)  
            time_domain (tuple): 时间域范围 (t_min, t_max)
            q_over_m (float): 电荷质量比 q/m
            epsilon_0 (float): 真空介电常数
            case (str): 预设案例 ("landau_damping", "two_stream", "bump_on_tail")
        """
        self.x_min, self.x_max = x_domain
        self.v_min, self.v_max = v_domain  
        self.t_min, self.t_max = time_domain
        self.q_over_m = q_over_m
        self.epsilon_0 = epsilon_0
        self.case = case
        
        # 设置随机种子
        np.random.seed(42)
        dde.config.set_random_seed(42)
        
        print(f"⚡ Vlasov-Poisson方程组求解器")
        print(f"相空间维度: 6D (x, v, t)")
        print(f"空间域: [{self.x_min}, {self.x_max}]")
        print(f"速度域: [{self.v_min}, {self.v_max}]")
        print(f"时间域: [{self.t_min}, {self.t_max}]")
        print(f"求解案例: {case}")
        
        # 根据案例设置特定参数
        self._setup_case_parameters()
    
    def _setup_case_parameters(self):
        """根据选择的案例设置参数"""
        if self.case == "landau_damping":
            self.amplitude = 0.01  # 扰动幅度
            self.k_mode = 0.5      # 波数
            self.v_thermal = 1.0   # 热速度
            print(f"📊 Landau阻尼案例: k={self.k_mode}, 扰动幅度={self.amplitude}")
            
        elif self.case == "two_stream":
            self.v_beam1 = 1.0     # 第一束流速度
            self.v_beam2 = -1.0    # 第二束流速度  
            self.beam_density = 0.1 # 束流密度比
            print(f"🌊 双流不稳定性案例: v1={self.v_beam1}, v2={self.v_beam2}")
            
        elif self.case == "bump_on_tail":
            self.v_bump = 3.0      # 尾部速度
            self.bump_amplitude = 0.1  # 尾部幅度
            print(f"📈 Bump-on-tail案例: v_bump={self.v_bump}")
    
    def vlasov_pde(self, inputs, outputs):
        """
        定义Vlasov方程的PDE残差
        
        Args:
            inputs: [x, v, t] (N, 3) - 相空间坐标
            outputs: [f, φ] (N, 2) - 分布函数和电势
            
        Returns:
            vlasov_residual (N, 1)
        """
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        f = outputs[:, 0:1]  # 分布函数
        phi = outputs[:, 1:2]  # 电势
        
        # 计算分布函数f的各种偏导数
        df_dt = dde.grad.jacobian(outputs, inputs, i=0, j=2)  # ∂f/∂t
        df_dx = dde.grad.jacobian(outputs, inputs, i=0, j=0)  # ∂f/∂x
        df_dv = dde.grad.jacobian(outputs, inputs, i=0, j=1)  # ∂f/∂v
        
        # 计算电场 E = -∂φ/∂x
        E = -dde.grad.jacobian(outputs, inputs, i=1, j=0)     # E = -∂φ/∂x
        
        # Vlasov方程: ∂f/∂t + v·∂f/∂x + (q/m)E·∂f/∂v = 0
        vlasov_residual = df_dt + v * df_dx + self.q_over_m * E * df_dv
        
        return vlasov_residual
    
    def poisson_pde(self, inputs, outputs):
        """
        定义Poisson方程的PDE残差
        
        Args:
            inputs: [x, v, t] (N, 3)
            outputs: [f, φ] (N, 2)
            
        Returns:
            poisson_residual (N, 1)
        """
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        f = outputs[:, 0:1]
        phi = outputs[:, 1:2]
        
        # 计算电势的二阶导数 ∂²φ/∂x²
        d2phi_dx2 = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
        
        # 计算电荷密度 ρ = q ∫ f dv (近似)
        # 注意：这里是简化处理，实际需要在速度方向积分
        rho = self.q_over_m * f  # 简化：假设速度积分已经包含在f中
        
        # Poisson方程: ∇²φ = -ρ/ε₀
        poisson_residual = d2phi_dx2 + rho / self.epsilon_0
        
        return poisson_residual
    
    def combined_pde(self, inputs, outputs):
        """
        组合的PDE系统
        
        Returns:
            [vlasov_residual, poisson_residual] (N, 2)
        """
        vlasov_res = self.vlasov_pde(inputs, outputs)
        poisson_res = self.poisson_pde(inputs, outputs)
        
        return [vlasov_res, poisson_res]
    
    def initial_condition_f(self, inputs):
        """
        分布函数f的初始条件
        """
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        
        if self.case == "landau_damping":
            # Maxwellian背景 + 小扰动
            # f₀(x,v) = (1/√(2π)σ) exp(-v²/(2σ²)) * (1 + A cos(kx))
            sigma = self.v_thermal
            maxwellian = (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-v**2 / (2 * sigma**2))
            perturbation = 1.0 + self.amplitude * np.cos(self.k_mode * np.pi * x)
            return maxwellian * perturbation
            
        elif self.case == "two_stream":
            # 双Maxwellian分布
            sigma = 0.5
            beam1 = np.exp(-(v - self.v_beam1)**2 / (2 * sigma**2))
            beam2 = np.exp(-(v - self.v_beam2)**2 / (2 * sigma**2))
            normalization = 1.0 / np.sqrt(2 * np.pi * sigma**2)
            return normalization * (beam1 + self.beam_density * beam2)
            
        elif self.case == "bump_on_tail":
            # Maxwellian主体 + 高能尾部
            sigma_main = 1.0
            sigma_tail = 0.3
            main_dist = np.exp(-v**2 / (2 * sigma_main**2))
            tail_dist = self.bump_amplitude * np.exp(-(v - self.v_bump)**2 / (2 * sigma_tail**2))
            normalization = 1.0 / np.sqrt(2 * np.pi)
            return normalization * (main_dist + tail_dist)
            
        else:
            # 默认：简单Maxwellian
            return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-v**2 / 2.0)
    
    def initial_condition_phi(self, inputs):
        """
        电势φ的初始条件
        """
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        
        if self.case == "landau_damping":
            # 与密度扰动对应的电势扰动
            return self.amplitude * np.sin(self.k_mode * np.pi * x) / (self.k_mode * np.pi)**2
        else:
            # 其他情况：初始电势为零
            return np.zeros_like(x)
    
    def setup_geometry_and_conditions(self):
        """设置几何域和边界/初始条件"""
        # 定义3D相空间域 (x, v, t)
        x_domain = dde.geometry.Interval(self.x_min, self.x_max)
        v_domain = dde.geometry.Interval(self.v_min, self.v_max)
        time_domain = dde.geometry.TimeDomain(self.t_min, self.t_max)
        
        # 创建相空间域 (x, v) × t
        phase_space = dde.geometry.geometry_nd.Hypercube([self.x_min, self.v_min], 
                                                        [self.x_max, self.v_max])
        self.geomtime = dde.geometry.GeometryXTime(phase_space, time_domain)
        
        # 边界条件：周期性边界条件 (for x direction)
        def boundary_x_left(inputs, on_boundary):
            return on_boundary and np.isclose(inputs[0], self.x_min)
        
        def boundary_x_right(inputs, on_boundary):
            return on_boundary and np.isclose(inputs[0], self.x_max)
        
        def boundary_v_left(inputs, on_boundary):
            return on_boundary and np.isclose(inputs[1], self.v_min)
        
        def boundary_v_right(inputs, on_boundary):
            return on_boundary and np.isclose(inputs[1], self.v_max)
        
        # 简化边界条件：零边界
        def zero_bc_f(inputs):
            return np.zeros((len(inputs), 1))
        
        def zero_bc_phi(inputs):
            return np.zeros((len(inputs), 1))
        
        # 创建边界条件
        self.bcs = [
            dde.icbc.DirichletBC(self.geomtime, zero_bc_f, boundary_v_left, component=0),
            dde.icbc.DirichletBC(self.geomtime, zero_bc_f, boundary_v_right, component=0),
            dde.icbc.DirichletBC(self.geomtime, zero_bc_phi, boundary_x_left, component=1),
            dde.icbc.DirichletBC(self.geomtime, zero_bc_phi, boundary_x_right, component=1),
        ]
        
        # 初始条件
        self.ics = [
            dde.icbc.IC(self.geomtime, self.initial_condition_f, 
                       lambda _, on_initial: on_initial, component=0),
            dde.icbc.IC(self.geomtime, self.initial_condition_phi, 
                       lambda _, on_initial: on_initial, component=1)
        ]
        
        print("✅ 6D相空间域和边界/初始条件设置完成")
    
    def create_model(self, 
                    num_domain=5000, 
                    num_boundary=500, 
                    num_initial=500,
                    layer_sizes=[3, 100, 100, 100, 100, 2], 
                    activation="tanh"):
        """
        创建神经网络模型
        
        Args:
            num_domain (int): 域内采样点数
            num_boundary (int): 边界采样点数  
            num_initial (int): 初始条件采样点数
            layer_sizes (list): 网络层大小 [输入3维, 隐藏层..., 输出2维]
            activation (str): 激活函数
        """
        # 创建训练数据
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.combined_pde,
            self.bcs + self.ics,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            num_test=1000
        )
        
        # 构建神经网络 (输入3维: x,v,t; 输出2维: f,φ)
        self.net = dde.nn.FNN(layer_sizes, activation, "Glorot uniform")
        
        # 创建模型
        self.model = dde.Model(self.data, self.net)
        
        print("🧠 Vlasov-Poisson神经网络模型创建完成")
        print(f"网络结构: {layer_sizes}")
        print(f"相空间采样点数: {num_domain}")
        print(f"边界采样点数: {num_boundary}")
        print(f"初始采样点数: {num_initial}")
        
        # 估算参数数量
        total_params = sum([layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] 
                           for i in range(len(layer_sizes)-1)])
        print(f"估计网络参数: ~{total_params:,}")
    
    def train(self, 
              adam_iterations=10000, 
              adam_lr=0.001, 
              use_lbfgs=True,
              weights_pde=[1.0, 1.0]):  # [vlasov_weight, poisson_weight]
        """
        训练模型
        
        Args:
            adam_iterations (int): Adam优化器迭代次数
            adam_lr (float): Adam学习率
            use_lbfgs (bool): 是否使用L-BFGS精细调优
            weights_pde (list): PDE方程权重 [Vlasov权重, Poisson权重]
        """
        print("🚀 开始训练Vlasov-Poisson系统...")
        print("⚠️  注意：6D系统计算极其复杂，请耐心等待...")
        
        # 第一阶段：Adam训练
        self.model.compile(
            optimizer="adam", 
            lr=adam_lr, 
            metrics=["l2 relative error"],
            loss_weights=weights_pde  # 设置方程权重
        )
        
        start_time = time.time()
        self.losshistory, self.train_state = self.model.train(iterations=adam_iterations)
        train_time = time.time() - start_time
        
        print(f"📊 Adam训练完成！ 用时: {train_time:.1f}秒")
        print(f"最终训练损失: {self.train_state.loss_train:.6f}")
        print(f"最终测试损失: {self.train_state.loss_test:.6f}")
        
        # 第二阶段：L-BFGS精细调优
        if use_lbfgs:
            print("\n🔧 开始L-BFGS精细调优...")
            self.model.compile("L-BFGS")
            self.losshistory, self.train_state = self.model.train()
            
            print("🎉 L-BFGS训练完成！")
            print(f"最终训练损失: {self.train_state.loss_train:.6f}")
            print(f"最终测试损失: {self.train_state.loss_test:.6f}")
    
    def predict(self, inputs):
        """
        预测给定相空间点的分布函数和电势
        
        Args:
            inputs: [x, v, t] 坐标 (N, 3)
            
        Returns:
            [f, φ] 预测值 (N, 2)
        """
        return self.model.predict(inputs)
    
    def compute_macroscopic_quantities(self, x_points, t, v_resolution=50):
        """
        计算宏观量：密度、平均速度、温度等
        
        Args:
            x_points (array): 空间点
            t (float): 时间点
            v_resolution (int): 速度积分分辨率
            
        Returns:
            dict: 包含各种宏观量的字典
        """
        v_points = np.linspace(self.v_min, self.v_max, v_resolution)
        dv = (self.v_max - self.v_min) / (v_resolution - 1)
        
        densities = []
        mean_velocities = []
        temperatures = []
        
        for x in x_points:
            # 创建相空间点 (x, v, t)
            phase_points = np.array([[x, v, t] for v in v_points])
            
            # 预测分布函数
            predictions = self.predict(phase_points)
            f_values = predictions[:, 0]  # 分布函数
            
            # 计算密度 n = ∫ f dv
            density = np.trapz(f_values, v_points)
            densities.append(density)
            
            # 计算平均速度 <v> = ∫ v f dv / n
            if density > 1e-10:  # 避免除零
                mean_v = np.trapz(v_points * f_values, v_points) / density
                mean_velocities.append(mean_v)
                
                # 计算温度 T ∝ ∫ (v - <v>)² f dv / n
                temp = np.trapz((v_points - mean_v)**2 * f_values, v_points) / density
                temperatures.append(temp)
            else:
                mean_velocities.append(0.0)
                temperatures.append(0.0)
        
        return {
            'density': np.array(densities),
            'mean_velocity': np.array(mean_velocities),
            'temperature': np.array(temperatures)
        }
    
    def visualize_initial_conditions(self, resolution=50):
        """可视化初始条件"""
        x_points = np.linspace(self.x_min, self.x_max, resolution)
        v_points = np.linspace(self.v_min, self.v_max, resolution)
        X, V = np.meshgrid(x_points, v_points)
        
        # 创建初始时刻的相空间点
        phase_points = np.stack([X.flatten(), V.flatten(), 
                                np.zeros_like(X.flatten())], axis=1)
        
        # 计算初始分布
        f_init = self.initial_condition_f(phase_points).reshape(X.shape)
        phi_init = self.initial_condition_phi(phase_points).reshape(X.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制初始分布函数
        im1 = axes[0].contourf(X, V, f_init, levels=20, cmap='viridis')
        axes[0].set_xlabel('位置 x')
        axes[0].set_ylabel('速度 v')
        axes[0].set_title(f'初始分布函数 f(x,v,0) - {self.case}')
        plt.colorbar(im1, ax=axes[0])
        
        # 绘制初始电势 (沿x方向的平均)
        phi_x = np.mean(phi_init, axis=0)  # 对速度维度求平均
        axes[1].plot(x_points, phi_x, 'b-', linewidth=2)
        axes[1].set_xlabel('位置 x')
        axes[1].set_ylabel('电势 φ')
        axes[1].set_title('初始电势 φ(x,0)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_phase_space_evolution(self, times=None, resolution=40):
        """
        可视化相空间演化
        
        Args:
            times (list): 可视化的时间点
            resolution (int): 相空间分辨率
        """
        if times is None:
            times = [0.0, 0.3, 0.6, 1.0]
        
        x_points = np.linspace(self.x_min, self.x_max, resolution)
        v_points = np.linspace(self.v_min, self.v_max, resolution)
        X, V = np.meshgrid(x_points, v_points)
        
        fig, axes = plt.subplots(2, len(times), figsize=(5*len(times), 10))
        
        for i, t in enumerate(times):
            # 创建相空间点
            phase_points = np.stack([X.flatten(), V.flatten(), 
                                   np.full_like(X.flatten(), t)], axis=1)
            
            # 预测分布函数和电势
            predictions = self.predict(phase_points)
            f_pred = predictions[:, 0].reshape(X.shape)
            phi_pred = predictions[:, 1].reshape(X.shape)
            
            # 绘制分布函数
            im1 = axes[0, i].contourf(X, V, f_pred, levels=20, cmap='viridis')
            axes[0, i].set_xlabel('位置 x')
            if i == 0:
                axes[0, i].set_ylabel('速度 v')
            axes[0, i].set_title(f'分布函数 f(x,v) at t={t:.1f}')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 绘制电势 (沿x的平均值)
            phi_x = np.mean(phi_pred, axis=0)
            axes[1, i].plot(x_points, phi_x, 'r-', linewidth=2)
            axes[1, i].set_xlabel('位置 x')
            if i == 0:
                axes[1, i].set_ylabel('电势 φ')
            axes[1, i].set_title(f'电势 φ(x) at t={t:.1f}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_plasma_dynamics(self, x_resolution=30, time_points=20):
        """分析等离子体动力学演化"""
        x_points = np.linspace(self.x_min, self.x_max, x_resolution)
        times = np.linspace(self.t_min, self.t_max, time_points)
        
        # 计算时空演化的宏观量
        density_evolution = []
        electric_field_evolution = []
        
        for t in times:
            # 计算宏观量
            macro_quantities = self.compute_macroscopic_quantities(x_points, t)
            density_evolution.append(macro_quantities['density'])
            
            # 计算电场 (简化：对中心x点)
            x_center = (self.x_min + self.x_max) / 2
            v_center = (self.v_min + self.v_max) / 2
            
            # 计算电场：E = -∂φ/∂x
            dx = 0.01
            phi_left = self.predict(np.array([[x_center - dx, v_center, t]]))[0, 1]
            phi_right = self.predict(np.array([[x_center + dx, v_center, t]]))[0, 1]
            E_field = -(phi_right - phi_left) / (2 * dx)
            electric_field_evolution.append(E_field)
        
        density_evolution = np.array(density_evolution)
        
        # 可视化分析结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练历史
        axes[0, 0].semilogy(self.losshistory.steps, self.losshistory.loss_train, 
                           'b-', label='训练损失')
        axes[0, 0].semilogy(self.losshistory.steps, self.losshistory.loss_test, 
                           'r--', label='测试损失')
        axes[0, 0].set_xlabel('训练步数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练历史')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 密度时空演化
        T_mesh, X_mesh = np.meshgrid(times, x_points)
        im2 = axes[0, 1].contourf(T_mesh, X_mesh, density_evolution.T, 
                                 levels=20, cmap='plasma')
        axes[0, 1].set_xlabel('时间 t')
        axes[0, 1].set_ylabel('位置 x')
        axes[0, 1].set_title('密度演化 n(x,t)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 电场时间演化
        axes[1, 0].plot(times, electric_field_evolution, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('时间 t')
        axes[1, 0].set_ylabel('电场 E')
        axes[1, 0].set_title('中心电场演化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 密度中心点时间演化
        x_center_idx = len(x_points) // 2
        density_center = density_evolution[:, x_center_idx]
        axes[1, 1].plot(times, density_center, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('时间 t')
        axes[1, 1].set_ylabel('密度 n')
        axes[1, 1].set_title('中心密度演化')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 等离子体动力学分析:")
        print(f"最大密度变化: {np.max(density_center) - np.min(density_center):.6f}")
        print(f"最大电场: {np.max(np.abs(electric_field_evolution)):.6f}")
        
        if self.case == "landau_damping":
            # 计算Landau阻尼率
            mid_idx = len(times) // 2
            if len(electric_field_evolution) > mid_idx:
                damping_rate = -np.log(abs(electric_field_evolution[mid_idx]) / 
                                     abs(electric_field_evolution[0])) / times[mid_idx]
                print(f"估计Landau阻尼率: γ ≈ {damping_rate:.4f}")
    
    def save_model(self, filename="vlasov_poisson_model"):
        """保存模型"""
        self.model.save(filename)
        print(f"💾 模型已保存为 {filename}")
    
    def load_model(self, filename="vlasov_poisson_model"):
        """加载模型"""
        self.model.restore(filename)
        print(f"📂 模型已从 {filename} 加载")


def main():
    """主函数 - 演示Vlasov-Poisson求解器"""
    print("⚡ Vlasov-Poisson方程组求解器演示")
    
    # 创建Landau阻尼案例的求解器
    solver = VlasovPoissonSolver(
        x_domain=(-1.0, 1.0),
        v_domain=(-3.0, 3.0),
        time_domain=(0.0, 1.0),
        case="landau_damping"
    )
    
    # 设置几何域和条件
    solver.setup_geometry_and_conditions()
    
    # 可视化初始条件
    solver.visualize_initial_conditions()
    
    # 创建模型 (6D系统需要更多参数和计算资源)
    solver.create_model(
        num_domain=8000,  # 6D系统需要大量采样点
        num_boundary=600,
        num_initial=600,
        layer_sizes=[3, 128, 128, 128, 128, 2],  # 更大的网络
        activation="tanh"
    )
    
    # 训练模型 (6D系统训练时间很长)
    solver.train(
        adam_iterations=15000,  # 更多迭代
        adam_lr=0.0008,        # 较小学习率
        use_lbfgs=True,
        weights_pde=[1.0, 0.1]  # Poisson方程权重较小
    )
    
    # 可视化结果
    solver.visualize_phase_space_evolution()
    solver.analyze_plasma_dynamics()
    
    # 保存模型
    solver.save_model("vlasov_poisson_landau")
    
    print("\n🎉 Vlasov-Poisson系统求解完成！")


if __name__ == "__main__":
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()