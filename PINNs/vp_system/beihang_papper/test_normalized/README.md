# test_normalized

本目录 `test_normalized` 包含用于测试归一化流程与训练/评估结果的文件，通常包含训练日志和可视化输出。

文件说明：

- `events.out.tfevents.*`：TensorBoard 事件文件，用于记录训练过程中的标量、图像与其他指标。

如何查看与使用：

1. 安装依赖（若尚未安装）：

   pip install -r ../../requirements.txt

2. 在项目根目录运行 TensorBoard 查看训练日志：

   tensorboard --logdir PINNs/vp_system/beihang_papper/test_normalized --port 6006

3. 若需复现结果，请参照上层的训练脚本 `beihang_papper/main.py` 并确保数据和配置一致。

注意：

- TensorBoard 文件可能较大，通常不直接纳入版本控制；如需我可以将其加入 `.gitignore` 或迁移至外部存储（如 S3）。
- 如需我添加更多示例或运行脚本，以及推送提交到远端，请告知。