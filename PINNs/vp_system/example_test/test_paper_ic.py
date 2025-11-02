"""
测试论文中的双流不稳定性初始条件
Test the two-stream instability initial condition from paper
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')

from config import get_configuration, use_ic_preset

# 使用论文中的初始条件
use_ic_preset('two_stream_paper')
config = get_configuration()

print("=" * 70)
print("Testing Paper Initial Condition (eq 6.7)")
print("=" * 70)
print(f"Parameters:")
print(f"  v_0 = {config['paper_v0']}")
print(f"  α = {config['paper_alpha']}")
print(f"  ω = {config['paper_omega']}")
print(f"  Domain: x ∈ [0, {config['x_max']:.3f}], v ∈ [±{config['v_max']}]")
print("=" * 70)

# 创建网格
nx, nv = 200, 200
x = torch.linspace(0, config['x_max'], nx)
v = torch.linspace(-config['v_max'], config['v_max'], nv)
X, V = torch.meshgrid(x, v, indexing='ij')

# 计算初始条件
v0 = config['paper_v0']
alpha = config['paper_alpha']
omega = config['paper_omega']

# 归一化因子
norm_factor = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))

# 两个高斯束
term1 = np.exp(-((V.numpy() - v0)**2) / 2.0)
term2 = np.exp(-((V.numpy() + v0)**2) / 2.0)

# 空间扰动
spatial_factor = 1.0 + alpha * np.cos(omega * X.numpy())

# 完整分布函数
f0 = norm_factor * (term1 + term2) * spatial_factor

# 创建图形
fig = plt.figure(figsize=(16, 10))

# 1. 相空间分布 f(x,v)
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.pcolormesh(X.numpy(), V.numpy(), f0, shading='auto', cmap='viridis')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('v', fontsize=12)
ax1.set_title(f'Phase Space: f(0,x,v)\nTwo-Stream (Paper eq 6.7)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='f(0,x,v)')
ax1.grid(True, alpha=0.3)

# 2. x=0 处的速度分布
ax2 = plt.subplot(2, 3, 2)
f_v_x0 = f0[0, :]
ax2.plot(v.numpy(), f_v_x0, 'b-', linewidth=2, label='x=0')
# 添加 x=L_x/2 处的分布
f_v_xmid = f0[nx//2, :]
ax2.plot(v.numpy(), f_v_xmid, 'r--', linewidth=2, label=f'x={config["x_max"]/2:.1f}')
ax2.set_xlabel('v', fontsize=12)
ax2.set_ylabel('f(v)', fontsize=12)
ax2.set_title('Velocity Distribution at Different x', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 密度分布 n(x) = ∫f dv
ax3 = plt.subplot(2, 3, 3)
n_x = np.trapz(f0, v.numpy(), axis=1)
ax3.plot(x.numpy(), n_x, 'g-', linewidth=2)
ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='n=1 (equilibrium)')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('n(x)', fontsize=12)
ax3.set_title('Density Profile: n(x) = ∫f dv', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 沿 v=v_0 的切片
ax4 = plt.subplot(2, 3, 4)
v_idx = np.argmin(np.abs(v.numpy() - v0))
f_x_v0 = f0[:, v_idx]
ax4.plot(x.numpy(), f_x_v0, 'm-', linewidth=2)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel(f'f(x, v={v0})', fontsize=12)
ax4.set_title(f'Spatial Slice at v = v₀ = {v0}', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. 沿 v=-v_0 的切片
ax5 = plt.subplot(2, 3, 5)
v_idx_neg = np.argmin(np.abs(v.numpy() + v0))
f_x_v0_neg = f0[:, v_idx_neg]
ax5.plot(x.numpy(), f_x_v0_neg, 'c-', linewidth=2)
ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel(f'f(x, v={-v0})', fontsize=12)
ax5.set_title(f'Spatial Slice at v = -v₀ = {-v0}', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 统计信息
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
Initial Condition Statistics
{'='*40}

Parameters:
  v₀ = {v0}
  α = {alpha}
  ω = {omega}
  
Domain:
  x ∈ [0, {config['x_max']:.3f}]
  v ∈ [{-config['v_max']}, {config['v_max']}]
  Period: 2π/ω = {2*np.pi/omega:.3f}
  
Normalization:
  Total mass: {np.trapz(n_x, x.numpy()):.6f}
  (should be ≈ L_x = {config['x_max']:.3f})
  
Distribution Properties:
  max(f): {f0.max():.6f}
  min(f): {f0.min():.6f}
  max(n): {n_x.max():.6f}
  min(n): {n_x.min():.6f}
  Density variation: {(n_x.max() - n_x.min()):.6f}
  
Beam Separation:
  Δv = 2v₀ = {2*v0}
  
Perturbation:
  Amplitude: {alpha * 100:.2f}%
  Wavelength: {2*np.pi/omega:.3f}
"""
ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('test_paper_initial_condition.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: test_paper_initial_condition.png")
print("\nVisualization completed!")
plt.show()

# 验证归一化
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
mass_per_unit_length = np.trapz(n_x, x.numpy()) / config['x_max']
print(f"Average density (should be ≈ 1.0): {mass_per_unit_length:.6f}")
print(f"Density perturbation: {(n_x.max() - n_x.min()) / mass_per_unit_length * 100:.4f}%")
print(f"Perturbation amplitude α: {alpha * 100:.3f}%")
print("\n✓ Verification completed!")
