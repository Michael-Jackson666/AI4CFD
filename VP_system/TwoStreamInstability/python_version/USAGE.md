# 🌀 双流不稳定性模拟 - 使用指南

## ✅ 你已经完成了什么

1. ✓ 编译了初始条件生成器 (`main`)
2. ✓ 生成了初始条件文件 (`initial.inp`)
3. ✓ 运行了简化的 VP 求解器
4. ✓ 生成了动画和诊断图

## 📁 生成的文件

```
TwoStreamInstability/
├── main                              # 编译好的可执行文件
├── initial.inp                       # 初始条件数据
├── two_stream_instability.gif        # 相空间演化动画 ⭐
├── diagnostics.png                   # 物理量诊断图 ⭐
│
├── quick_start.py                    # 快速启动脚本
├── simple_vp_solver.py              # 简化 VP 求解器
├── run_simulation.py                 # 完整运行脚本
└── README.md                         # 详细文档
```

## 🎯 如何查看双涡旋演化

### 方法 1: 查看刚刚生成的结果（推荐开始这里）

```bash
# 查看动画
open two_stream_instability.gif

# 查看诊断图
open diagnostics.png
```

### 方法 2: 运行更长时间的模拟（观察完整演化）

**我已经将 `solver.inp` 中的 `n_iter` 从 25 改为 100**，现在重新运行：

```bash
# 重新运行求解器（会模拟到 t=20）
python3 simple_vp_solver.py

# 查看新的结果
open two_stream_instability.gif
open diagnostics.png
```

**预期看到的现象**：
- **t=0-5**: 初始的双 Maxwellian 分布，轻微扰动
- **t=5-10**: 双涡旋结构开始形成（线性不稳定性增长）
- **t=10-15**: 双涡旋结构清晰可见，达到最大强度
- **t=15-20**: 非线性饱和，涡旋可能合并或重新分布

## 📊 理解结果

### 动画说明 (two_stream_instability.gif)

动画包含 4 个子图：

1. **左上 - 相空间 (Phase Space)**
   - 横轴 x: 空间位置
   - 纵轴 v: 速度
   - 颜色: 分布函数 f(x,v) 的值
   - **双涡旋**: 看起来像两个旋转的"眼睛"

2. **右上 - 空间密度 (Spatial Density)**
   - 电子密度随空间的分布
   - 应该看到周期性的密度聚集

3. **左下 - 电场 (Electric Field)**
   - 电场强度随空间的变化
   - 初期增长，后期振荡

4. **右下 - 速度分布 (Velocity Distribution)**
   - 在 x=L/2 处的速度分布
   - 显示两束电子的分布演化

### 诊断图说明 (diagnostics.png)

1. **电场能量演化**
   - 应该看到指数增长（线性阶段）
   - 然后饱和或振荡（非线性阶段）

2. **动能演化**
   - 与电场能量互补
   - 能量在电场和粒子间转换

3. **质量守恒**
   - 应该保持接近常数
   - 偏差反映数值误差

4. **最终相空间**
   - 展示最后时刻的双涡旋结构

## 🔧 参数调整指南

### 1. 观察不同时间尺度的演化

编辑 `solver.inp`:
```
n_iter  50    # 短时间 (t=10)
n_iter  100   # 中等时间 (t=20) ← 当前设置
n_iter  200   # 长时间 (t=40)
```

然后重新运行：
```bash
python3 simple_vp_solver.py
```

### 2. 改变物理参数

编辑 `main.c` 中的参数：

```c
double T = 1.0;        // 改为 0.5 (冷束流, 更不稳定)
                       // 改为 2.0 (热束流, 更稳定)

double k = 1.0;        // 改为 0.5 (长波扰动)
                       // 改为 2.0 (短波扰动)

// 在第57行修改束流速度
double temp2 = exp(- (v[j] - 2.0) * ...);  // 改为 3.0 (更快)
double temp3 = exp(- (v[j] + 2.0) * ...);  // 改为 3.0 (更快)
```

修改后需要重新编译和运行：
```bash
gcc -o main main.c -lm
./main
python3 simple_vp_solver.py
```

### 3. 提高网格分辨率

编辑 `solver.inp`:
```
size  256 256    # 更高分辨率（但计算更慢）
```

然后：
```bash
./main  # 重新生成初始条件
python3 simple_vp_solver.py
```

## 🎬 高级使用

### 生成高质量视频（需要 ffmpeg）

修改 `simple_vp_solver.py`，在保存动画部分添加：

```python
# 在 anim.save(...) 后添加
try:
    anim.save('two_stream_instability.mp4', writer='ffmpeg', 
              fps=10, dpi=150, bitrate=1800)
    print("   ✓ 高清视频已保存: two_stream_instability.mp4")
except:
    print("   (ffmpeg 未安装，跳过 MP4 保存)")
```

### 导出数据用于其他分析

在 `simple_vp_solver.py` 末尾添加：

```python
# 保存数据
np.savez('simulation_data.npz',
         times=times,
         x=x,
         v=v,
         f_history=f_history,
         E_history=E_history)
print("✓ 数据已保存: simulation_data.npz")
```

然后在其他 Python 脚本中加载：
```python
import numpy as np
data = np.load('simulation_data.npz', allow_pickle=True)
times = data['times']
f_history = data['f_history']
# ... 进行自定义分析
```

## 📖 推荐阅读

### 理解双流不稳定性

1. **线性理论**
   - 增长率: γ ≈ √(3/2) k v_th
   - 不稳定条件: 两束速度差 > 3 v_th

2. **非线性演化**
   - 相空间涡旋形成（trapping）
   - 能量级联
   - 最终达到稳态

### 相关物理现象

- **Landau 阻尼**: 波与粒子共振吸收能量
- **等离子体振荡**: 电子集体运动
- **相空间混合**: 精细结构（filamentation）

## 🚀 下一步

### 尝试不同的初始条件

1. **单束流 + 扰动**
   ```c
   // 修改 main.c 第56-57行
   f[p] = (1.0 + 0.1 * temp1) * temp2 / temp4;  // 只保留一个束流
   ```

2. **三束流**
   ```c
   double temp5 = exp(- v[j] * v[j] / (2.0 * T));
   f[p] = ... + temp5 / temp4;  // 添加第三个束流
   ```

### 使用完整的 HyPar 求解器

如果你安装了 HyPar：

```bash
# 运行 HyPar（会生成 op_*.dat 文件）
HyPar

# 使用 gnuplot 生成动画
gnuplot plot.gp

# 或使用完整的 Python 脚本
python3 run_simulation.py
```

HyPar 的优势：
- 更高精度（5阶 WENO）
- 更稳定
- 支持并行计算

### 连接到 DeepONet 学习

你现在生成的数据可以用于训练 DeepONet！

```bash
# 切换到 DeepONet 目录
cd ../DeepONet/vp_system

# 使用这里的数据作为参考
# 可以修改 data_generate.py 使用相同的参数
```

## 🐛 故障排查

### 问题 1: 动画中有数值不稳定（NaN 值）

**解决方案**：
- 减小时间步长：`dt = 0.1`
- 减小扰动幅度：在 `main.c` 中改 `0.1` 为 `0.05`

### 问题 2: 看不清楚双涡旋

**解决方案**：
- 增加模拟时间：`n_iter = 150`
- 提高网格分辨率：`size 256 256`
- 调整颜色映射：在 `simple_vp_solver.py` 中改 `cmap='jet'` 为 `cmap='viridis'`

### 问题 3: 程序运行太慢

**解决方案**：
- 降低网格分辨率：`size 64 64`
- 减少时间步数：`n_iter = 50`
- 使用更粗的时间步长（慎用）：`dt = 0.5`

## 📞 需要帮助？

查看详细文档：
```bash
cat README.md
```

或查看单个文件的说明：
```bash
python3 quick_start.py  # 环境检查和指导
```

---

**祝你观察到清晰的双涡旋现象！🌀🌀**

如果你看到了相空间中两个旋转的"眼睛"，那就是双流不稳定性的标志性特征！
