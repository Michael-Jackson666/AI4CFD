# --- Corrected Gnuplot Script ---

# 设置输出为 gif 动画，每帧之间延迟100毫秒 (0.1秒)
set term gif animate delay 10
set output 'op.gif'

# 设置绘图视角和样式
set view map
set size square
set palette defined ( 0 '#000090',\
                      1 '#0000b0',\
                      2 '#0000d0',\
                      3 '#0020ff',\
                      4 '#00a0ff',\
                      5 '#00d0ff',\
                      6 '#20ffff',\
                      7 '#50ffff',\
                      8 '#70ffff',\
                      9 '#90ffff',\
                      10 '#b0ffff',\
                      11 '#ffffff',\
                      12 '#ffffb0',\
                      13 '#ffff90',\
                      14 '#ffff70',\
                      15 '#ffff50',\
                      16 '#ffff20',\
                      17 '#ffd000',\
                      18 '#ffb000',\
                      19 '#ff9000',\
                      20 '#ff7000',\
                      21 '#ff5000',\
                      22 '#ff2000',\
                      23 '#d00000',\
                      24 '#b00000',\
                      25 '#900000' )

# 直接根据 solver.inp 中的设置进行循环
# 从 0 开始，到 20000 结束，每一步增加 500
# 这将正确地匹配文件名 op_00000.dat, op_00500.dat, ...
do for [timestep=0:20000:500] {
    filename = sprintf("op_%05d.dat", timestep)
    # splot 用于绘制3D数据（这里是2D平面上的值）
    # u 1:2:3 表示使用第1、2、3列数据作为 x, y, z
    splot filename u 1:2:3 with pm3d title sprintf("Time step %d", timestep)
}

# 结束输出
unset output

print "GIF animation 'op.gif' has been generated successfully."