# inital_overfix

本目录包含训练过程中的结果可视化图像（按 epoch 保存），用于展示模型随训练迭代的演化效果。

文件说明：

- `results_epoch_500.png`、`results_epoch_1000.png`、...、`results_epoch_5000.png`：不同时刻（训练轮次）的结果可视化。

如何复现/生成这些图像：

1. 在仓库根目录或合适的虚拟环境中安装依赖：

   pip install -r requirements.txt

2. 从本目录执行训练或复现脚本（参见上层 `main.py`）：

   python ../main.py

3. 训练脚本运行完成后，输出的可视化图像会保存在此目录。

注意事项：

- 若图像较大或包含模型权重等大文件，建议使用 Git LFS 管理或将权重移至外部存储并将文件名记录在仓库中。
- 如需将这些结果自动化生成到不同文件夹，请查看 `beihang_papper/main.py` 中的保存路径设置并据此调整。

如需进一步完善（添加示例图示、命令行参数说明，或将大文件加入 `.gitignore`），告诉我即可。