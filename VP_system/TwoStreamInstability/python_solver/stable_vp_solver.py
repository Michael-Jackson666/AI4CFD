#!/usr/bin/env python3
"""
稳定的Vlasov-Poisson求解器 (基于HyPar配置)
使用5阶WENO空间离散 + RK4时间积分
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft, ifft, fftfreq

class StableVPSolver:
    def __init__(self, nx, nv, Lx, Lv, dt, T, epsilon=1e-10):
        """
        参数:
            nx, nv: 空间和速度网格点数
            Lx, Lv: 物理域大小
            dt: 时间步长
            T: 总时间
            epsilon: 数值耗散系数
        """
        self.nx, self.nv = nx, nv
        self.Lx, self.Lv = Lx, Lv
        self.dt, self.T = dt, T
        self.epsilon = epsilon
        
        # 网格
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.v = np.linspace(-Lv/2, Lv/2, nv, endpoint=False)
        self.dx = Lx / nx
        self.dv = Lv / nv
        
        # FFT频率（用于泊松方程）
        self.kx = 2 * np.pi * fftfreq(nx, self.dx)
        self.kx[0] = 1  # 避免除零
        
        # 存储
        self.f = None
        self.E = None
        self.diagnostics = {'time': [], 'energy': [], 'mass': [], 'entropy': []}
        
    def load_initial_condition(self, filename='initial.inp'):
        """从HyPar格式的二进制文件读取初始条件"""
        try:
            # HyPar格式: 4字节整数(ndims) + 4字节整数×ndims(size) + 8字节双精度数据
            with open(filename, 'rb') as f:
                ndims = np.fromfile(f, dtype=np.int32, count=1)[0]
                sizes = np.fromfile(f, dtype=np.int32, count=ndims)
                data = np.fromfile(f, dtype=np.float64)
            
            # 重塑数据
            self.f = data.reshape((self.nv, self.nx), order='C')
            print(f"✓ 从 {filename} 加载初始条件")
            print(f"  网格: {self.nx}×{self.nv}, 数据形状: {self.f.shape}")
            print(f"  f范围: [{self.f.min():.3e}, {self.f.max():.3e}]")
            
        except FileNotFoundError:
            print(f"✗ 找不到 {filename}, 使用默认初始条件")
            self.set_default_initial_condition()
            
    def set_default_initial_condition(self):
        """双Maxwellian初始条件"""
        X, V = np.meshgrid(self.x, self.v, indexing='ij')
        X, V = X.T, V.T  # 转置以匹配 (nv, nx)
        
        v1, v2 = 2.0, -2.0  # 束流速度
        T = 1.0
        amplitude = 0.1
        k = 1.0
        
        # 双Maxwellian
        M1 = (2*np.pi*T)**(-0.5) * np.exp(-0.5*(V-v1)**2/T)
        M2 = (2*np.pi*T)**(-0.5) * np.exp(-0.5*(V-v2)**2/T)
        
        self.f = 0.5*(M1 + M2) * (1 + amplitude*np.cos(k*X))
        
    def compute_electric_field(self):
        """通过泊松方程计算电场: -∂ₓE = ρ - 1"""
        # 密度: ρ(x) = ∫ f dv
        rho = np.trapz(self.f, self.v, axis=0)
        
        # 傅里叶空间求解: ρ̂ = ikₓ Ê
        rho_k = fft(rho - 1)
        E_k = -1j * rho_k / self.kx
        E_k[0] = 0  # 周期边界条件
        
        self.E = np.real(ifft(E_k))
        
    def weno5_flux(self, f, vel):
        """5阶WENO重构通量（单向）"""
        n = f.shape[0]
        flux = np.zeros_like(f)
        
        # 正速度方向
        if vel >= 0:
            for i in range(n):
                # 5点模板
                im2 = (i-2) % n
                im1 = (i-1) % n
                i0 = i
                ip1 = (i+1) % n
                ip2 = (i+2) % n
                
                # ENO候选
                f1 = (1/3)*f[im2] - (7/6)*f[im1] + (11/6)*f[i0]
                f2 = -(1/6)*f[im1] + (5/6)*f[i0] + (1/3)*f[ip1]
                f3 = (1/3)*f[i0] + (5/6)*f[ip1] - (1/6)*f[ip2]
                
                # 光滑指标
                IS1 = (13/12)*(f[im2]-2*f[im1]+f[i0])**2 + 0.25*(f[im2]-4*f[im1]+3*f[i0])**2
                IS2 = (13/12)*(f[im1]-2*f[i0]+f[ip1])**2 + 0.25*(f[im1]-f[ip1])**2
                IS3 = (13/12)*(f[i0]-2*f[ip1]+f[ip2])**2 + 0.25*(3*f[i0]-4*f[ip1]+f[ip2])**2
                
                # 非线性权重
                eps = 1e-6
                alpha1 = 0.1 / (eps + IS1)**2
                alpha2 = 0.6 / (eps + IS2)**2
                alpha3 = 0.3 / (eps + IS3)**2
                w1 = alpha1 / (alpha1 + alpha2 + alpha3)
                w2 = alpha2 / (alpha1 + alpha2 + alpha3)
                w3 = alpha3 / (alpha1 + alpha2 + alpha3)
                
                flux[i] = w1*f1 + w2*f2 + w3*f3
        else:
            # 负速度方向（镜像）
            for i in range(n):
                im2 = (i+2) % n
                im1 = (i+1) % n
                i0 = i
                ip1 = (i-1) % n
                ip2 = (i-2) % n
                
                f1 = (1/3)*f[im2] - (7/6)*f[im1] + (11/6)*f[i0]
                f2 = -(1/6)*f[im1] + (5/6)*f[i0] + (1/3)*f[ip1]
                f3 = (1/3)*f[i0] + (5/6)*f[ip1] - (1/6)*f[ip2]
                
                IS1 = (13/12)*(f[im2]-2*f[im1]+f[i0])**2 + 0.25*(f[im2]-4*f[im1]+3*f[i0])**2
                IS2 = (13/12)*(f[im1]-2*f[i0]+f[ip1])**2 + 0.25*(f[im1]-f[ip1])**2
                IS3 = (13/12)*(f[i0]-2*f[ip1]+f[ip2])**2 + 0.25*(3*f[i0]-4*f[ip1]+f[ip2])**2
                
                eps = 1e-6
                alpha1 = 0.1 / (eps + IS1)**2
                alpha2 = 0.6 / (eps + IS2)**2
                alpha3 = 0.3 / (eps + IS3)**2
                w1 = alpha1 / (alpha1 + alpha2 + alpha3)
                w2 = alpha2 / (alpha1 + alpha2 + alpha3)
                w3 = alpha3 / (alpha1 + alpha2 + alpha3)
                
                flux[i] = w1*f1 + w2*f2 + w3*f3
                
        return flux
    
    def rhs(self, f):
        """计算右端项: -v∂ₓf - E∂ᵥf"""
        dfdt = np.zeros_like(f)
        
        # 计算电场
        rho = np.trapz(f, self.v, axis=0)
        rho_k = fft(rho - 1)
        E_k = -1j * rho_k / self.kx
        E_k[0] = 0
        E = np.real(ifft(E_k))
        
        # 空间平流: -v∂ₓf (逐v层处理)
        for j in range(self.nv):
            v_j = self.v[j]
            flux = self.weno5_flux(f[j, :], v_j)
            dfdt[j, :] -= v_j * (np.roll(flux, -1) - flux) / self.dx
            
        # 速度平流: -E∂ᵥf (逐x层处理)
        for i in range(self.nx):
            E_i = E[i]
            flux = self.weno5_flux(f[:, i], -E_i)  # 注意符号
            dfdt[:, i] -= E_i * (np.roll(flux, -1) - flux) / self.dv
            
        # 数值耗散
        dfdt -= self.epsilon * f
        
        return dfdt
    
    def step_rk4(self):
        """RK4时间积分"""
        k1 = self.rhs(self.f)
        k2 = self.rhs(self.f + 0.5*self.dt*k1)
        k3 = self.rhs(self.f + 0.5*self.dt*k2)
        k4 = self.rhs(self.f + self.dt*k3)
        
        self.f += (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 非负性保持
        self.f = np.maximum(self.f, 0)
        
    def compute_diagnostics(self, t):
        """计算守恒量"""
        # 质量
        mass = np.trapz(np.trapz(self.f, self.v, axis=0), self.x)
        
        # 能量: ∫∫ v²f/2 dx dv + ∫ E²/2 dx
        rho = np.trapz(self.f, self.v, axis=0)
        rho_k = fft(rho - 1)
        E_k = -1j * rho_k / self.kx
        E_k[0] = 0
        E = np.real(ifft(E_k))
        
        kinetic = 0.5 * np.trapz(np.trapz(self.v**2 * self.f.T, self.v), self.x)
        electric = 0.5 * np.trapz(E**2, self.x)
        energy = kinetic + electric
        
        # 熵
        f_safe = self.f + 1e-16
        entropy = -np.trapz(np.trapz(self.f * np.log(f_safe), self.v, axis=0), self.x)
        
        self.diagnostics['time'].append(t)
        self.diagnostics['mass'].append(mass)
        self.diagnostics['energy'].append(energy)
        self.diagnostics['entropy'].append(entropy)
        
    def solve(self, save_interval=None):
        """主求解循环"""
        if self.f is None:
            self.load_initial_condition()
            
        n_steps = int(self.T / self.dt)
        if save_interval is None:
            save_interval = max(1, n_steps // 100)
            
        snapshots = []
        
        print(f"\n开始求解:")
        print(f"  时间步: {n_steps}, dt={self.dt}, T={self.T}")
        print(f"  网格: {self.nx}×{self.nv}")
        print(f"  保存间隔: 每 {save_interval} 步")
        
        for step in range(n_steps + 1):
            t = step * self.dt
            
            if step % save_interval == 0:
                self.compute_diagnostics(t)
                snapshots.append((t, self.f.copy()))
                print(f"  步 {step:4d} / {n_steps} | t={t:5.2f} | "
                      f"Mass={self.diagnostics['mass'][-1]:.6f} | "
                      f"Energy={self.diagnostics['energy'][-1]:.6f}")
                
            if step < n_steps:
                self.step_rk4()
                
        print("✓ 求解完成!\n")
        return snapshots
    
    def plot_results(self, snapshots, filename='two_stream_result.png'):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 相空间演化
        ax = axes[0, 0]
        for i, idx in enumerate([0, len(snapshots)//3, 2*len(snapshots)//3, -1]):
            t, f = snapshots[idx]
            extent = [self.x[0], self.x[-1], self.v[0], self.v[-1]]
            im = ax.contourf(self.x, self.v, f, levels=20, cmap='viridis', alpha=0.6)
            ax.set_title(f'Phase Space (t={t:.1f})')
            ax.set_xlabel('x')
            ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        # 2. 最终相空间
        ax = axes[0, 1]
        t_final, f_final = snapshots[-1]
        im = ax.contourf(self.x, self.v, f_final, levels=50, cmap='viridis')
        ax.set_title(f'Final Phase Space (t={t_final:.1f})')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        plt.colorbar(im, ax=ax)
        
        # 3. 守恒量
        ax = axes[1, 0]
        times = self.diagnostics['time']
        ax.plot(times, self.diagnostics['mass'], label='Mass', lw=2)
        ax.plot(times, self.diagnostics['energy'], label='Energy', lw=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Conservation Laws')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 密度演化
        ax = axes[1, 1]
        for i, idx in enumerate([0, len(snapshots)//2, -1]):
            t, f = snapshots[idx]
            rho = np.trapz(f, self.v, axis=0)
            ax.plot(self.x, rho, label=f't={t:.1f}', lw=2)
        ax.set_xlabel('x')
        ax.set_ylabel('ρ(x)')
        ax.set_title('Density Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ 结果图保存到: {filename}")
        
    def create_animation(self, snapshots, filename='two_stream.gif'):
        """创建动画"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t0, f0 = snapshots[0]
        vmin, vmax = f0.min(), f0.max()
        
        im = ax.contourf(self.x, self.v, f0, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='f(x,v)')
        ax.set_xlabel('Position x', fontsize=12)
        ax.set_ylabel('Velocity v', fontsize=12)
        title = ax.set_title(f'Two-Stream Instability | t = 0.00', fontsize=14)
        
        def animate(i):
            t, f = snapshots[i]
            ax.clear()
            im = ax.contourf(self.x, self.v, f, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_ylabel('Velocity v', fontsize=12)
            ax.set_title(f'Two-Stream Instability | t = {t:.2f}', fontsize=14)
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=100, blit=False)
        anim.save(filename, writer='pillow', fps=10, dpi=100)
        print(f"✓ 动画保存到: {filename}")
        plt.close()

if __name__ == '__main__':
    # 参数（与HyPar配置一致）
    nx, nv = 128, 128
    Lx, Lv = 2*np.pi, 12.0
    dt = 0.1  # 减小时间步长以提高稳定性
    T = 20.0
    
    # 创建求解器
    solver = StableVPSolver(nx, nv, Lx, Lv, dt, T, epsilon=1e-4)
    
    # 加载初始条件（会自动查找initial.inp）
    solver.load_initial_condition()
    
    # 求解
    snapshots = solver.solve(save_interval=10)
    
    # 可视化
    solver.plot_results(snapshots, 'two_stream_stable.png')
    solver.create_animation(snapshots, 'two_stream_stable.gif')
    
    print("\n全部完成! 文件:")
    print("  - two_stream_stable.png")
    print("  - two_stream_stable.gif")
