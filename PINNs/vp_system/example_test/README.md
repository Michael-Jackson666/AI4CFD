# example_test

本目录包含用于验证初始条件与基准测试的测试脚本与资源。

文件说明：

- `PAPER_IC_GUIDE.md`：关于论文中初始条件的说明与可视化参考。
- `setup_paper_ic.py`：用于设置论文中初始条件的辅助脚本。
- `test_initial_conditions.py`、`test_paper_ic.py`：用于运行单元/回归测试的脚本。
- 其它 PNG 文件：可视化参考图像。

如何运行测试：

1. 安装依赖：

   pip install -r ../../requirements.txt

2. 运行单个测试脚本：

   python test_initial_conditions.py

   或

   python test_paper_ic.py

3. （可选）使用 pytest 运行：

   pytest -q

注意：某些测试可能依赖于特定的 Python 环境或外部数值库，运行前请确保环境正确配置。问题请参考 `PAPER_IC_GUIDE.md` 或联系维护者。