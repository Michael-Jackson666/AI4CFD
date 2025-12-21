# local_10000

本目录（`local_10000`）通常用于保存本地实验结果或配置，尤其是在尝试较大样本/网格（例如 10000）时的输出数据。

文件说明（可能存在的文件）：

- 模型输出图像（如 `results_epoch_*.png`）
- 训练或测试的日志文件（如 `train.log`、`metrics.json`）
- 生成的模型权重（建议使用 Git LFS 或外部存储）

使用说明：

1. 请在合适的 Python 环境中安装依赖：

   pip install -r ../../requirements.txt

2. 若要运行或复现实验，请参考上层 `main.py` 或本仓库中相应的训练脚本，示例命令：

   python ../main.py --config local_10000

注意事项：

- 请避免将大文件（如权重、数据）直接提交到 Git 仓库；如需我可以帮你添加 `.gitignore` 或设置 Git LFS。
- 如果你希望我创建一些示例文件或将已有输出移动到该目录，告诉我我可以代为处理。