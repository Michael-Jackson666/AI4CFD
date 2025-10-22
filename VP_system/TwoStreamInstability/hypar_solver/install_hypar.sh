#!/bin/bash
#
# HyPar 安装脚本 - macOS/Linux
# 自动下载、编译并安装 HyPar 求解器
#

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "HyPar 求解器安装脚本"
echo "========================================================================"
echo ""

# 检查系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macOS"
    echo "检测到系统: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SYSTEM="Linux"
    echo "检测到系统: Linux"
else
    echo "不支持的操作系统: $OSTYPE"
    exit 1
fi

# 检查必要工具
echo ""
echo "步骤 1: 检查必要工具..."
echo "----------------------------------------"

# 检查 git
if ! command -v git &> /dev/null; then
    echo "✗ Git 未安装"
    echo "请先安装 Git:"
    if [ "$SYSTEM" == "macOS" ]; then
        echo "  xcode-select --install"
    else
        echo "  sudo apt-get install git"
    fi
    exit 1
fi
echo "✓ Git 已安装: $(git --version)"

# 检查编译器
if ! command -v gcc &> /dev/null; then
    echo "✗ GCC 未安装"
    echo "请先安装编译器"
    exit 1
fi
echo "✓ GCC 已安装: $(gcc --version | head -n 1)"

# 检查 make
if ! command -v make &> /dev/null; then
    echo "✗ Make 未安装"
    exit 1
fi
echo "✓ Make 已安装: $(make --version | head -n 1)"

# 创建临时目录
INSTALL_DIR="$HOME/hypar_install"
echo ""
echo "步骤 2: 创建安装目录..."
echo "----------------------------------------"
echo "安装目录: $INSTALL_DIR"

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# 下载 HyPar
echo ""
echo "步骤 3: 下载 HyPar 源代码..."
echo "----------------------------------------"

if [ -d "hypar" ]; then
    echo "HyPar 目录已存在，更新代码..."
    cd hypar
    git pull
else
    echo "克隆 HyPar 仓库..."
    git clone https://github.com/debog/hypar.git
    cd hypar
fi

echo "✓ HyPar 源代码下载完成"

# 配置
echo ""
echo "步骤 4: 配置编译选项..."
echo "----------------------------------------"

# 检查是否需要 autoreconf
if [ ! -f "configure" ]; then
    echo "运行 autoreconf 生成 configure 脚本..."
    if command -v autoreconf &> /dev/null; then
        autoreconf -i
    else
        echo "警告: autoreconf 未安装，尝试直接配置..."
    fi
fi

# 配置（串行版本）
echo "配置 HyPar（串行版本）..."
if [ -f "configure" ]; then
    ./configure --prefix=$HOME/.local
else
    echo "警告: configure 脚本不存在，将尝试直接编译..."
fi

echo "✓ 配置完成"

# 编译
echo ""
echo "步骤 5: 编译 HyPar..."
echo "----------------------------------------"
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

# 尝试使用 Makefile
if [ -f "Makefile" ]; then
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
else
    echo "尝试使用简单编译方式..."
    # 简单编译方式
    cd src
    gcc -c *.c -I../include -O3
    gcc -o ../bin/HyPar *.o -lm
    cd ..
fi

echo "✓ 编译完成"

# 安装
echo ""
echo "步骤 6: 安装 HyPar..."
echo "----------------------------------------"

# 创建目标目录
mkdir -p $HOME/.local/bin

# 查找编译好的可执行文件
if [ -f "bin/HyPar" ]; then
    cp bin/HyPar $HOME/.local/bin/
    echo "✓ HyPar 已安装到: $HOME/.local/bin/HyPar"
elif [ -f "src/HyPar" ]; then
    cp src/HyPar $HOME/.local/bin/
    echo "✓ HyPar 已安装到: $HOME/.local/bin/HyPar"
else
    # 尝试使用 make install
    if [ -f "Makefile" ]; then
        make install
        echo "✓ HyPar 已通过 make install 安装"
    else
        echo "✗ 找不到编译好的 HyPar 可执行文件"
        exit 1
    fi
fi

# 设置权限
chmod +x $HOME/.local/bin/HyPar

# 添加到 PATH
echo ""
echo "步骤 7: 配置环境变量..."
echo "----------------------------------------"

# 检查 shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

# 添加 PATH
if ! grep -q ".local/bin" $SHELL_RC; then
    echo "" >> $SHELL_RC
    echo "# HyPar 路径" >> $SHELL_RC
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $SHELL_RC
    echo "✓ 已添加 PATH 到 $SHELL_RC"
    echo "  请运行: source $SHELL_RC"
else
    echo "✓ PATH 已配置"
fi

# 临时添加到当前 PATH
export PATH="$HOME/.local/bin:$PATH"

# 验证安装
echo ""
echo "步骤 8: 验证安装..."
echo "----------------------------------------"

if command -v HyPar &> /dev/null; then
    echo "✓ HyPar 安装成功！"
    echo ""
    HyPar -version 2>&1 || echo "HyPar 位置: $(which HyPar)"
else
    echo "✗ HyPar 未在 PATH 中找到"
    echo "  手动添加到 PATH:"
    echo "    export PATH=\"$HOME/.local/bin:\$PATH\""
fi

# 清理
echo ""
echo "步骤 9: 清理..."
echo "----------------------------------------"
echo "安装文件保留在: $INSTALL_DIR"
echo "如果需要，可以手动删除: rm -rf $INSTALL_DIR"

echo ""
echo "========================================================================"
echo "✓ HyPar 安装完成！"
echo "========================================================================"
echo ""
echo "下一步:"
echo "  1. 重新加载 shell 配置:"
echo "     source $SHELL_RC"
echo ""
echo "  2. 验证安装:"
echo "     HyPar -version"
echo ""
echo "  3. 运行模拟:"
echo "     cd /Users/jack/Desktop/ML/AI4CFD/VP_system/TwoStreamInstability"
echo "     ./main"
echo "     HyPar"
echo ""
