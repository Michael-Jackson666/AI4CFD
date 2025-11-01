"""
Vlasov-Poisson 方程求解器 - Landau阻尼现象
使用半拉格朗日方法求解1D1V Vlasov-Poisson系统

物理方程:
∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0  (Vlasov方程)
∂E/∂x = ∫f dv - n0             (Poisson方程)

其中:
f(x,v,t) - 分布函数
E(x,t)   - 电场
n0       - 背景密度
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import fftpack


class VlasovPoissonSolver:
    """Vlasov-Poisson方程求解器"""
    
    def __init__(self, nx=64, nv=64, Lx=4*np.pi, Lv=6.0, dt=0.1):
        """
        初始化求解器
        
        参数:
            nx: x方向网格点数
            nv: v方向网格点数
            Lx: x方向区域长度
            Lv: v方向区域长度（-Lv/2 到 Lv/2）
            dt: 时间步长
        """
        self.nx = nx
        self.nv = nv
        self.Lx = Lx
        self.Lv = Lv
        self.dt = dt
        
        # 创建网格
        self.dx = Lx / nx
        self.dv = Lv / nv
        
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.v = np.linspace(-Lv/2, Lv/2, nv, endpoint=False)
        
        self.X, self.V = np.meshgrid(self.x, self.v, indexing='ij')
        
        # 波数（用于谱方法求解Poisson方程）
        self.kx = 2 * np.pi * fftpack.fftfreq(nx, d=self.dx)
        
        # 初始化
        self.f = None  # 分布函数
        self.E = None  # 电场
        self.rho = None  # 电荷密度
        
        # 用于存储历史数据
        self.history = {
            'time': [],
            'energy': [],
            'electric_energy': [],
            'kinetic_energy': [],
            'L2_norm': [],
            'electric_field_mode': []
        }
    
    def initialize_landau_damping(self, k=0.5, alpha=0.01, v_thermal=1.0):
        """
        初始化Landau阻尼测试问题
        
        初始条件:
        f(x,v,0) = (1/√(2π)vth) * exp(-v²/(2vth²)) * (1 + α*cos(kx))
        
        参数:
            k: 扰动波数
            alpha: 扰动幅度
            v_thermal: 热速度
        """
        # Maxwell分布
        f_maxwell = (1.0 / (np.sqrt(2*np.pi) * v_thermal)) * \
                    np.exp(-self.V**2 / (2 * v_thermal**2))
        
        # 加入空间扰动
        self.f = f_maxwell * (1.0 + alpha * np.cos(k * self.X))
        
        # 计算初始电场
        self.update_fields()
        
        print(f"初始化Landau阻尼问题:")
        print(f"  波数 k = {k}")
        print(f"  扰动幅度 α = {alpha}")
        print(f"  热速度 vth = {v_thermal}")
        print(f"  网格: nx={self.nx}, nv={self.nv}")
        print(f"  区域: x∈[0,{self.Lx:.2f}], v∈[{-self.Lv/2:.2f},{self.Lv/2:.2f}]")
        print(f"  时间步长: dt={self.dt}")
    
    def update_fields(self):
        """更新电荷密度和电场（使用谱方法求解Poisson方程）"""
        # 计算电荷密度：ρ = ∫f dv - 1
        self.rho = np.trapz(self.f, self.v, axis=1) - 1.0
        
        # 使用FFT求解Poisson方程: ∂E/∂x = ρ
        # 在傅里叶空间: ikE_k = ρ_k
        rho_k = fftpack.fft(self.rho)
        
        # 避免k=0的奇异性
        E_k = np.zeros_like(rho_k, dtype=complex)
        E_k[1:] = -1j * rho_k[1:] / self.kx[1:]
        E_k[0] = 0.0  # 平均电场为0
        
        # 逆FFT得到物理空间的电场
        self.E = np.real(fftpack.ifft(E_k))
    
    def advect_x(self, dt_step):
        """
        x方向对流: ∂f/∂t + v·∂f/∂x = 0
        使用半拉格朗日方法
        """
        # 计算特征线的起点
        x_star = (self.X - self.V * dt_step) % self.Lx
        
        # 使用样条插值
        interpolator = RectBivariateSpline(
            self.x, self.v, self.f, kx=3, ky=3
        )
        
        # 在新位置插值
        f_new = np.zeros_like(self.f)
        for i in range(self.nx):
            for j in range(self.nv):
                f_new[i, j] = interpolator(x_star[i, j], self.v[j], grid=False)
        
        self.f = f_new
    
    def advect_v(self, dt_step):
        """
        v方向对流: ∂f/∂t + E·∂f/∂v = 0
        使用半拉格朗日方法
        """
        # 电场扩展到速度方向
        E_extended = self.E[:, np.newaxis]
        
        # 计算特征线的起点
        v_star = self.V - E_extended * dt_step
        
        # 边界处理：限制在速度范围内
        v_star = np.clip(v_star, self.v[0], self.v[-1])
        
        # 使用样条插值
        interpolator = RectBivariateSpline(
            self.x, self.v, self.f, kx=3, ky=3
        )
        
        # 在新位置插值
        f_new = np.zeros_like(self.f)
        for i in range(self.nx):
            for j in range(self.nv):
                f_new[i, j] = interpolator(self.x[i], v_star[i, j], grid=False)
        
        self.f = f_new
    
    def step(self):
        """
        单步时间推进（使用Strang分裂）
        Strang splitting: S = S_v(dt/2) ∘ S_x(dt) ∘ S_v(dt/2)
        """
        # v方向半步
        self.advect_v(self.dt / 2)
        self.update_fields()
        
        # x方向完整步
        self.advect_x(self.dt)
        
        # v方向半步
        self.update_fields()
        self.advect_v(self.dt / 2)
        
        # 最终场更新
        self.update_fields()
    
    def compute_diagnostics(self):
        """计算诊断量"""
        # 电场能量: E_e = (1/2)∫E² dx
        electric_energy = 0.5 * np.trapz(self.E**2, self.x)
        
        # 动能: E_k = (1/2)∫∫v²f dv dx
        kinetic_energy = 0.5 * np.trapz(
            np.trapz(self.V**2 * self.f, self.v, axis=1), 
            self.x
        )
        
        # 总能量
        total_energy = electric_energy + kinetic_energy
        
        # L2范数
        L2_norm = np.sqrt(np.trapz(np.trapz(self.f**2, self.v, axis=1), self.x))
        
        # 电场第一模式的幅值（用于观察Landau阻尼）
        E_fft = fftpack.fft(self.E)
        E_mode_1 = np.abs(E_fft[1])
        
        return {
            'total_energy': total_energy,
            'electric_energy': electric_energy,
            'kinetic_energy': kinetic_energy,
            'L2_norm': L2_norm,
            'electric_field_mode': E_mode_1
        }
    
    def solve(self, T_final, save_interval=10):
        """
        求解VP系统
        
        参数:
            T_final: 最终时间
            save_interval: 保存间隔（每多少步保存一次）
        
        返回:
            history: 包含时间历史的字典
        """
        n_steps = int(T_final / self.dt)
        
        print(f"\n开始求解...")
        print(f"总步数: {n_steps}, 最终时间: {T_final}")
        print(f"保存间隔: {save_interval} 步")
        
        # 清空历史
        self.history = {
            'time': [],
            'energy': [],
            'electric_energy': [],
            'kinetic_energy': [],
            'L2_norm': [],
            'electric_field_mode': []
        }
        
        # 保存初始状态
        diagnostics = self.compute_diagnostics()
        self.history['time'].append(0.0)
        for key in diagnostics:
            self.history[key.replace('total_', '')].append(diagnostics[key])
        
        # 时间推进
        for step in range(n_steps):
            self.step()
            
            # 保存诊断量
            if (step + 1) % save_interval == 0:
                t = (step + 1) * self.dt
                diagnostics = self.compute_diagnostics()
                
                self.history['time'].append(t)
                for key in diagnostics:
                    self.history[key.replace('total_', '')].append(diagnostics[key])
                
                if (step + 1) % (save_interval * 10) == 0:
                    print(f"步数 {step+1}/{n_steps}, t={t:.2f}, "
                          f"E_field_mode={diagnostics['electric_field_mode']:.6e}")
        
        print("求解完成!")
        
        # 转换为numpy数组
        for key in self.history:
            self.history[key] = np.array(self.history[key])
        
        return self.history
    
    def get_state(self):
        """获取当前状态"""
        return {
            'f': self.f.copy(),
            'E': self.E.copy(),
            'rho': self.rho.copy(),
            'x': self.x,
            'v': self.v
        }


def compute_landau_damping_rate(k, v_thermal=1.0):
    """
    计算理论Landau阻尼率
    
    对于Maxwell分布: γ ≈ -√(π/8) * (ωp/k³vth³) * exp(-1/(2k²vth²))
    其中 ωp = 1 (等离子体频率)
    
    简化形式: γ ≈ -√(π/8) * exp(-1/(2k²vth²) - 3/2) / k²
    """
    gamma = -np.sqrt(np.pi / 8) * np.exp(-1.0 / (2 * k**2 * v_thermal**2) - 1.5) / k**2
    return gamma


if __name__ == "__main__":
    print("Vlasov-Poisson求解器测试")
    print("=" * 50)
    
    # 创建求解器
    solver = VlasovPoissonSolver(
        nx=64,
        nv=64,
        Lx=4*np.pi,
        Lv=6.0,
        dt=0.1
    )
    
    # 初始化Landau阻尼问题
    k_wave = 0.5
    solver.initialize_landau_damping(k=k_wave, alpha=0.01, v_thermal=1.0)
    
    # 计算理论阻尼率
    gamma_theory = compute_landau_damping_rate(k_wave, v_thermal=1.0)
    print(f"\n理论Landau阻尼率: γ = {gamma_theory:.6f}")
    
    # 求解
    history = solver.solve(T_final=50.0, save_interval=5)
    
    print(f"\n能量守恒检查:")
    energy_variation = np.std(history['energy']) / np.mean(history['energy'])
    print(f"  能量相对变化: {energy_variation:.2e}")
