#!/usr/bin/env python3
"""
双流不稳定性模拟运行脚本
自动化运行 C 程序、HyPar 求解器，并生成可视化结果
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import struct
import glob

# 颜色配置
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 10)

def compile_initial_condition_generator():
    """编译初始条件生成器"""
    print("="*70)
    print("步骤 1: 编译初始条件生成器")
    print("="*70)
    
    if os.path.exists('main'):
        print("✓ 可执行文件 'main' 已存在")
        return True
    
    print("编译 main.c ...")
    result = subprocess.run(['gcc', '-o', 'main', 'main.c', '-lm'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ 编译成功")
        return True
    else:
        print("✗ 编译失败:")
        print(result.stderr)
        return False

def generate_initial_condition():
    """运行初始条件生成器"""
    print("\n" + "="*70)
    print("步骤 2: 生成初始条件")
    print("="*70)
    
    if not os.path.exists('solver.inp'):
        print("✗ 错误: solver.inp 不存在")
        return False
    
    print("运行 ./main 生成 initial.inp ...")
    result = subprocess.run(['./main'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ 初始条件生成成功")
        print(result.stdout)
        return True
    else:
        print("✗ 生成失败:")
        print(result.stderr)
        return False

def check_hypar():
    """检查 HyPar 是否安装"""
    result = subprocess.run(['which', 'HyPar'], capture_output=True, text=True)
    return result.returncode == 0

def run_hypar_simulation():
    """运行 HyPar 求解器"""
    print("\n" + "="*70)
    print("步骤 3: 运行 HyPar 求解器")
    print("="*70)
    
    if not check_hypar():
        print("✗ 错误: HyPar 未安装或不在 PATH 中")
        print("请先安装 HyPar: https://github.com/debog/hypar")
        return False
    
    print("运行 HyPar 模拟...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    
    # 运行 HyPar 并实时显示输出
    process = subprocess.Popen(['HyPar'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1,
                              universal_newlines=True)
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✓ HyPar 模拟完成")
        return True
    else:
        print("\n✗ HyPar 运行失败")
        return False

def read_binary_output(filename, nx, nv):
    """读取二进制输出文件"""
    try:
        with open(filename, 'rb') as f:
            # 读取网格
            x = np.array(struct.unpack('d' * nx, f.read(8 * nx)))
            v = np.array(struct.unpack('d' * nv, f.read(8 * nv)))
            
            # 读取分布函数
            f_data = np.array(struct.unpack('d' * (nx * nv), f.read(8 * nx * nv)))
            f_data = f_data.reshape((nv, nx))
            
        return x, v, f_data
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")
        return None, None, None

def create_phase_space_animation():
    """创建相空间动画"""
    print("\n" + "="*70)
    print("步骤 4: 创建相空间动画")
    print("="*70)
    
    # 从 solver.inp 读取网格大小
    nx, nv = 128, 128  # 默认值
    
    try:
        with open('solver.inp', 'r') as f:
            for line in f:
                if 'size' in line:
                    parts = line.split()
                    nx = int(parts[1])
                    nv = int(parts[2])
                    break
    except:
        print("警告: 无法从 solver.inp 读取网格大小，使用默认值 128x128")
    
    print(f"网格大小: {nx} × {nv}")
    
    # 查找所有输出文件
    output_files = sorted(glob.glob('op_*.dat'))
    
    if len(output_files) == 0:
        print("✗ 错误: 未找到输出文件 (op_*.dat)")
        return False
    
    print(f"找到 {len(output_files)} 个输出文件")
    
    # 读取所有数据
    frames_data = []
    times = []
    
    for i, filename in enumerate(output_files):
        # 从文件名提取时间步
        timestep = int(filename.split('_')[1].split('.')[0])
        
        x, v, f = read_binary_output(filename, nx, nv)
        
        if x is not None:
            frames_data.append((x, v, f))
            times.append(timestep)
            if i % 5 == 0:
                print(f"  加载文件 {i+1}/{len(output_files)}: {filename}")
    
    if len(frames_data) == 0:
        print("✗ 错误: 无法读取任何数据文件")
        return False
    
    print(f"✓ 成功加载 {len(frames_data)} 帧数据")
    
    # 创建动画
    print("\n创建动画...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 初始化第一帧
    x, v, f = frames_data[0]
    
    # 相空间图
    im1 = ax1.contourf(x, v, f, levels=50, cmap='jet')
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Velocity v', fontsize=12)
    ax1.set_title(f'Phase Space (t={times[0]*0.2:.2f})', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Distribution f(x,v)')
    
    # 密度图
    density = np.trapz(f, v, axis=0)
    line1, = ax2.plot(x, density, 'b-', linewidth=2)
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Density ρ(x)', fontsize=12)
    ax2.set_title(f'Spatial Density (t={times[0]*0.2:.2f})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(density) * 1.2])
    
    plt.tight_layout()
    
    def update(frame):
        """更新动画帧"""
        x, v, f = frames_data[frame]
        t = times[frame] * 0.2  # 假设 dt=0.2
        
        # 更新相空间图
        ax1.clear()
        im = ax1.contourf(x, v, f, levels=50, cmap='jet')
        ax1.set_xlabel('Position x', fontsize=12)
        ax1.set_ylabel('Velocity v', fontsize=12)
        ax1.set_title(f'Phase Space (t={t:.2f})', fontsize=14)
        
        # 更新密度图
        density = np.trapz(f, v, axis=0)
        line1.set_ydata(density)
        ax2.set_title(f'Spatial Density (t={t:.2f})', fontsize=14)
        
        return [im, line1]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(frames_data), 
                        interval=200, blit=False, repeat=True)
    
    # 保存动画
    print("保存动画到 two_stream_instability.gif ...")
    anim.save('two_stream_instability.gif', writer='pillow', fps=5, dpi=100)
    print("✓ 动画已保存: two_stream_instability.gif")
    
    # 也保存为 MP4（如果有 ffmpeg）
    try:
        print("尝试保存为 MP4 格式...")
        anim.save('two_stream_instability.mp4', writer='ffmpeg', fps=5, dpi=150)
        print("✓ 视频已保存: two_stream_instability.mp4")
    except:
        print("  (ffmpeg 未安装，跳过 MP4 保存)")
    
    plt.close()
    return True

def create_diagnostic_plots():
    """创建诊断图：电场、能量等"""
    print("\n" + "="*70)
    print("步骤 5: 创建诊断图")
    print("="*70)
    
    # 从 solver.inp 读取网格大小
    nx, nv = 128, 128
    
    try:
        with open('solver.inp', 'r') as f:
            for line in f:
                if 'size' in line:
                    parts = line.split()
                    nx = int(parts[1])
                    nv = int(parts[2])
                    break
    except:
        pass
    
    # 查找所有输出文件
    output_files = sorted(glob.glob('op_*.dat'))
    
    if len(output_files) == 0:
        print("未找到输出文件")
        return False
    
    # 计算物理量
    times = []
    electric_energy = []
    kinetic_energy = []
    total_mass = []
    
    print("计算物理量...")
    
    for filename in output_files:
        timestep = int(filename.split('_')[1].split('.')[0])
        x, v, f = read_binary_output(filename, nx, nv)
        
        if x is not None:
            dt = 0.2  # 从 solver.inp
            t = timestep * dt
            times.append(t)
            
            # 计算密度
            density = np.trapz(f, v, axis=0)
            
            # 电场能量 (简化计算)
            rho = density - 1.0
            E_energy = np.trapz(rho**2, x)
            electric_energy.append(E_energy)
            
            # 动能
            V, X = np.meshgrid(v, x)
            K_energy = np.trapz(np.trapz(f * V**2, v, axis=0), x)
            kinetic_energy.append(K_energy)
            
            # 总质量（应该守恒）
            mass = np.trapz(density, x)
            total_mass.append(mass)
    
    print(f"✓ 处理了 {len(times)} 个时间步")
    
    # 绘制诊断图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 电场能量
    axes[0, 0].semilogy(times, electric_energy, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Electric Field Energy', fontsize=12)
    axes[0, 0].set_title('Electric Field Energy Evolution', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 动能
    axes[0, 1].plot(times, kinetic_energy, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel('Kinetic Energy', fontsize=12)
    axes[0, 1].set_title('Kinetic Energy Evolution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 总质量
    axes[1, 0].plot(times, total_mass, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Total Mass', fontsize=12)
    axes[1, 0].set_title('Mass Conservation Check', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最后时刻的相空间
    x, v, f = read_binary_output(output_files[-1], nx, nv)
    im = axes[1, 1].contourf(x, v, f, levels=50, cmap='jet')
    axes[1, 1].set_xlabel('Position x', fontsize=12)
    axes[1, 1].set_ylabel('Velocity v', fontsize=12)
    axes[1, 1].set_title(f'Final Phase Space (t={times[-1]:.2f})', fontsize=14)
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=150, bbox_inches='tight')
    print("✓ 诊断图已保存: diagnostics.png")
    
    plt.close()
    return True

def main():
    """主函数"""
    print("\n" + "="*70)
    print("双流不稳定性模拟 - 自动化运行脚本")
    print("="*70 + "\n")
    
    # 检查当前目录
    required_files = ['solver.inp', 'physics.inp', 'boundary.inp', 'main.c']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("✗ 错误: 缺少必要文件:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    print("✓ 所有必要文件已就绪\n")
    
    # 执行步骤
    if not compile_initial_condition_generator():
        return
    
    if not generate_initial_condition():
        return
    
    if not run_hypar_simulation():
        print("\n提示: 如果 HyPar 未安装，请参考:")
        print("  https://github.com/debog/hypar")
        return
    
    if not create_phase_space_animation():
        return
    
    if not create_diagnostic_plots():
        return
    
    print("\n" + "="*70)
    print("✓ 所有步骤完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - two_stream_instability.gif  (相空间演化动画)")
    print("  - two_stream_instability.mp4  (高清视频，如果有 ffmpeg)")
    print("  - diagnostics.png             (物理量诊断图)")
    print("\n你可以使用以下命令查看:")
    print("  open two_stream_instability.gif")
    print("  open diagnostics.png")
    print()

if __name__ == '__main__':
    main()
