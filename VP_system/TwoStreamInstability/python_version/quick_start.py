#!/usr/bin/env python3
"""
简化版运行脚本 - 检查环境并给出指导
"""

import os
import subprocess
import sys

def check_environment():
    """检查环境"""
    print("="*70)
    print("环境检查")
    print("="*70 + "\n")
    
    # 检查 GCC
    print("1. 检查 GCC 编译器...")
    result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        version = result.stdout.split('\n')[0]
        print(f"   ✓ GCC 已安装: {version}")
    else:
        print("   ✗ GCC 未安装")
        return False
    
    # 检查 HyPar
    print("\n2. 检查 HyPar 求解器...")
    result = subprocess.run(['which', 'HyPar'], capture_output=True, text=True)
    hypar_installed = result.returncode == 0
    
    if hypar_installed:
        print(f"   ✓ HyPar 已安装: {result.stdout.strip()}")
    else:
        print("   ✗ HyPar 未安装")
        print("   需要安装 HyPar 才能运行模拟")
        print("   安装方法: https://github.com/debog/hypar")
    
    # 检查 Python 库
    print("\n3. 检查 Python 库...")
    required_libs = ['numpy', 'matplotlib']
    all_installed = True
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"   ✓ {lib} 已安装")
        except ImportError:
            print(f"   ✗ {lib} 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n   安装缺失的库:")
        print("   pip install numpy matplotlib")
    
    # 检查必要文件
    print("\n4. 检查配置文件...")
    required_files = ['solver.inp', 'physics.inp', 'boundary.inp', 'main.c']
    all_present = True
    
    for f in required_files:
        if os.path.exists(f):
            print(f"   ✓ {f}")
        else:
            print(f"   ✗ {f} 缺失")
            all_present = False
    
    print("\n" + "="*70)
    
    return hypar_installed and all_installed and all_present

def print_instructions():
    """打印使用说明"""
    print("\n" + "="*70)
    print("使用说明")
    print("="*70 + "\n")
    
    print("要观察双流不稳定现象，请按以下步骤操作:\n")
    
    print("步骤 1: 编译初始条件生成器")
    print("  $ gcc -o main main.c -lm")
    print()
    
    print("步骤 2: 生成初始条件")
    print("  $ ./main")
    print("  这会生成 initial.inp 文件")
    print()
    
    print("步骤 3: 运行 HyPar 求解器")
    print("  $ HyPar")
    print("  这会生成一系列 op_xxxxx.dat 文件")
    print()
    
    print("步骤 4: 生成动画")
    print("  方法 A (使用我的 Python 脚本):")
    print("    $ python3 run_simulation.py")
    print()
    print("  方法 B (使用 gnuplot):")
    print("    $ gnuplot plot.gp")
    print("    这会生成 op.gif")
    print()
    
    print("="*70)
    print("\n提示: 如果想观察更长时间的演化，修改 solver.inp:")
    print("  n_iter    从 25 改为 100 或更大")
    print("  这样可以看到更完整的双涡旋演化过程")
    print()

def quick_run():
    """快速运行（如果环境满足）"""
    print("\n" + "="*70)
    print("快速运行")
    print("="*70 + "\n")
    
    # 编译
    if not os.path.exists('main'):
        print("编译 main.c ...")
        result = subprocess.run(['gcc', '-o', 'main', 'main.c', '-lm'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("编译失败:", result.stderr)
            return False
        print("✓ 编译成功")
    else:
        print("✓ 可执行文件已存在")
    
    # 生成初始条件
    print("\n生成初始条件...")
    result = subprocess.run(['./main'], capture_output=True, text=True)
    if result.returncode != 0:
        print("生成失败:", result.stderr)
        return False
    print("✓ 初始条件生成成功")
    print(result.stdout)
    
    # 检查是否可以运行 HyPar
    result = subprocess.run(['which', 'HyPar'], capture_output=True, text=True)
    if result.returncode != 0:
        print("\n✗ HyPar 未安装，无法继续")
        print("请先安装 HyPar: https://github.com/debog/hypar")
        return False
    
    print("\n准备运行 HyPar...")
    print("这可能需要几分钟，请耐心等待...")
    
    # 询问用户
    response = input("\n是否继续运行 HyPar? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return False
    
    print("\n运行 HyPar...")
    result = subprocess.run(['HyPar'], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("HyPar 运行失败:", result.stderr)
        return False
    
    print("\n✓ HyPar 运行完成")
    
    # 检查输出文件
    import glob
    output_files = glob.glob('op_*.dat')
    print(f"✓ 生成了 {len(output_files)} 个输出文件")
    
    # 生成动画
    print("\n生成动画...")
    
    # 尝试使用 gnuplot
    result = subprocess.run(['which', 'gnuplot'], capture_output=True, text=True)
    if result.returncode == 0:
        print("使用 gnuplot 生成动画...")
        result = subprocess.run(['gnuplot', 'plot.gp'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 动画已生成: op.gif")
            print("\n查看动画:")
            print("  $ open op.gif")
        else:
            print("gnuplot 运行失败:", result.stderr)
    else:
        print("gnuplot 未安装，尝试使用 Python...")
        try:
            exec(open('run_simulation.py').read())
        except Exception as e:
            print(f"Python 脚本运行失败: {e}")
    
    return True

def main():
    print("\n" + "="*70)
    print("双流不稳定性模拟 - 快速启动脚本")
    print("="*70)
    
    # 检查环境
    env_ready = check_environment()
    
    # 打印说明
    print_instructions()
    
    if env_ready:
        print("\n✓ 环境检查通过")
        response = input("\n是否立即运行模拟? (y/n): ")
        if response.lower() == 'y':
            quick_run()
    else:
        print("\n✗ 环境未就绪，请先安装缺失的组件")
        print("\n如果只是想编译和生成初始条件，可以运行:")
        print("  $ gcc -o main main.c -lm")
        print("  $ ./main")

if __name__ == '__main__':
    main()
