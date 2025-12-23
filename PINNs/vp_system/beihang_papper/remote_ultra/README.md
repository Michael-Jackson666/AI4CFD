# remote_ultra

此目录（`remote_ultra`）用于存放远程实验或超大规模实验的产出与辅助文件，适用于需要通过远程计算资源执行的大型训练或推理任务。

可能包含的内容：

- 训练/推理结果图、分析图像（如 `results_epoch_*.png`）
- 日志与度量文件（`train.log`、`metrics.json`）
- 模型权重与检查点（建议使用 Git LFS 或外部对象存储，并在仓库中仅保留下载或引用说明）

如何使用：

1. 确保依赖已安装：

   pip install -r ../../requirements.txt

2. 如需从远程下载/上传数据或权重，请使用合适的存储（S3、GCS、或 SSH/scp）并在脚本中配置路径。

注意事项：

- 远程/超大文件通常不适合直接提交到 Git 仓库，建议使用 Git LFS 或将文件上传至外部存储并在仓库中保留引用或下载脚本。
- 如需，我可以为你添加 `.gitattributes` 配置或演示如何使用 Git LFS。