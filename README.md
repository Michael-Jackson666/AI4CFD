# AI4CFD

轻量版说明：本项目收集了用于求解偏微分方程（PDE）的若干深度学习方法实现与教程，包含示例代码、训练脚本与若干实用案例。

主要模块（简要）：

- PINNs — 物理信息神经网络（`PINNs/`）
- DeepONet — 深度算子网络（`DeepONet/`）
- FNO — 傅里叶神经算子（`FNO/`）
- TNN — 张量神经网络（`TNN/`）
- Transformer 相关实现（`Transformer/`）

快速开始：

```bash
git clone https://github.com/Michael-Jackson666/AI4CFD.git
cd AI4CFD
pip install -r requirements.txt
```

常用示例：
- PINNs 教程：`cd PINNs/tutorial && jupyter notebook tutorial_eng.ipynb`
- TNN 5D 示例：`cd TNN/train/dim5 && python ex_5_1_dim5.py`
- DeepONet 教程：`cd DeepONet/tutorial && jupyter notebook operator_learning_torch.ipynb`

依赖（简要）：Python >= 3.8，PyTorch >= 1.10，NumPy/SciPy/Matplotlib。

更多细节请参阅各子模块下的 `README.md`，许可：MIT。
