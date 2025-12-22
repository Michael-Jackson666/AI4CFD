# local_100000

本目录（`local_100000`）用于保存本地大规模实验输出与结果（例如在 100000 样本或网格下的训练/推理输出）。

常见内容（示例）：

- 训练/测试结果图像（`results_epoch_*.png`）
- 日志与度量文件（`train.log`、`metrics.json`）
- 模型权重（建议使用 Git LFS 或外部存储）

使用与复现：

1. 安装依赖：

   pip install -r ../../requirements.txt

2. 运行训练/评估脚本（参考上层 `main.py`）：

   python ../main.py --config local_100000

注意事项：

- 本目录可能包含较大的二进制或权重文件，建议将这些文件添加到 `.gitignore` 或使用 Git LFS 管理。需要的话，我可以为你设置 `.gitattributes` 与 Git LFS。 
- 如果你希望我将现有结果移动到该目录或添加演示脚本，我可以继续操作。