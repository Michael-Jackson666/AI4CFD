# true_test

本目录包含用于验证和分析 Vlasov–Poisson 问题的真实测试与结果文件，主要用于展示模型在真实场景下的表现（如 Landau 衰减）。

文件说明：

- `vlasov_poisson_solver.py`：用于运行或复现仿真结果的脚本。
- `vlasov_poisson_initial_conditions_landau_damping.png`：初始条件可视化图。
- `vlasov_poisson_evolution_landau_damping.png`、`vlasov_poisson_analysis_landau_damping.png`：演化与分析结果图。
- `vlasov_poisson_landau_damping_8000iter-8000.weights.h5`：训练产生的模型权重示例（注意：大文件，通常不应直接纳入版本控制）。

如何使用：

1. 确保已安装依赖：

   pip install -r ../../requirements.txt

2. 运行仿真/推理：

   python vlasov_poisson_solver.py

3. 若只想查看结果图像，可直接查看 PNG 文件。

注意事项：

- 权重文件较大，建议通过外部存储或 Git LFS 管理；若不希望将权重包含在版本库，请从仓库中移除并添加到 `.gitignore`。如需我为你设置 `.gitattributes` 或 `.gitignore`，告诉我即可。