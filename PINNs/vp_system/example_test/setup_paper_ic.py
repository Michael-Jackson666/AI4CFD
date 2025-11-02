"""
快速启动脚本 - 使用论文中的双流不稳定性初始条件
Quick start script for paper's two-stream instability IC

根据论文 equation (6.7):
f_0(x,v) = (1/4π)(1 + α cos(ω·x))[exp(-(v-v_0)²/2) + exp(-(v+v_0)²/2)]

参数:
- α = 0.003 (扰动幅度)
- ω = 0.2 (波数)
- v_0 = 2.4 (束流速度)
"""

from config import use_ic_preset, DOMAIN, INITIAL_CONDITION, TRAINING, LOGGING

# 使用论文中的初始条件
use_ic_preset('two_stream_paper')

# 可以根据需要调整这些参数
print("\n" + "=" * 70)
print("PAPER INITIAL CONDITION SETUP")
print("=" * 70)
print("\n当前配置:")
print(f"  初始条件类型: {INITIAL_CONDITION['type']}")
print(f"  v_0 (束流速度): {INITIAL_CONDITION['paper_v0']}")
print(f"  α (扰动幅度): {INITIAL_CONDITION['paper_alpha']}")
print(f"  ω (波数): {INITIAL_CONDITION['paper_omega']}")
print(f"\n域设置:")
print(f"  t_max: {DOMAIN['t_max']}")
print(f"  x_max: {DOMAIN['x_max']:.3f} (对应一个波长: 2π/ω = {2*3.14159/0.2:.3f})")
print(f"  v_max: {DOMAIN['v_max']}")
print(f"\n训练设置:")
print(f"  Epochs: {TRAINING['epochs']}")
print(f"  Learning rate: {TRAINING['learning_rate']}")
print(f"  采样点: PDE={TRAINING['n_pde']}, IC={TRAINING['n_ic']}, BC={TRAINING['n_bc']}")
print(f"\n输出目录:")
print(f"  {LOGGING['plot_dir']}")
print("\n" + "=" * 70)

# 提示用户
print("\n准备开始训练!")
print("\n运行以下命令:")
print("  python test_paper_ic.py    # 先可视化初始条件")
print("  python main.py             # 开始训练")
print("\n如需调整参数，请编辑 config.py 文件")
print("=" * 70)
