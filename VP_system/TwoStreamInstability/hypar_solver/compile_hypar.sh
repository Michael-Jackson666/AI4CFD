#!/bin/bash
#
# HyPar 简化编译脚本 - 手动编译方式
#

set -e

echo "========================================================================"
echo "HyPar 简化编译脚本"
echo "========================================================================"
echo ""

cd ~/hypar_install/hypar

echo "步骤 1: 安装 autotools（如果需要）..."
echo "----------------------------------------"

# 检查 autoconf
if ! command -v autoconf &> /dev/null; then
    echo "安装 autotools..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install autoconf automake libtool
        else
            echo "✗ 请先安装 Homebrew: https://brew.sh"
            echo "或手动安装 autotools"
            exit 1
        fi
    else
        # Linux
        sudo apt-get install -y autoconf automake libtool
    fi
fi

echo "✓ autotools 已就绪"

echo ""
echo "步骤 2: 生成 configure 脚本..."
echo "----------------------------------------"

autoreconf -i

echo "✓ configure 脚本生成完成"

echo ""
echo "步骤 3: 配置..."
echo "----------------------------------------"

./configure --prefix=$HOME/.local

echo "✓ 配置完成"

echo ""
echo "步骤 4: 编译..."
echo "----------------------------------------"

make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 2)

echo "✓ 编译完成"

echo ""
echo "步骤 5: 安装..."
echo "----------------------------------------"

make install

echo "✓ 安装完成"

# 添加到 PATH
echo ""
echo "步骤 6: 配置环境..."
echo "----------------------------------------"

SHELL_RC="$HOME/.zshrc"
if ! grep -q ".local/bin" $SHELL_RC; then
    echo "" >> $SHELL_RC
    echo "# HyPar 路径" >> $SHELL_RC
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $SHELL_RC
    echo "✓ 已添加到 $SHELL_RC"
fi

export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "========================================================================"
echo "✓ HyPar 编译安装完成！"
echo "========================================================================"
echo ""
echo "验证安装:"
echo "  source ~/.zshrc"
echo "  HyPar -version"
echo ""
