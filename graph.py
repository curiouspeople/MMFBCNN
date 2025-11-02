import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 生成离散数据示例
x = np.array([0,1, 2,3, 4, 5,6, 7,8,9, 10])  #
y = np.array([85,86,88,75,65,66,78,55,60,90,80])   # 对应的Y值

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制折线图（添加数据点标记）
ax.plot(x, y,
        marker='*',         # 圆形数据点标记
        markersize=8,       # 标记大小
        markerfacecolor='black',  # 标记填充色
        markeredgecolor='black',# 标记边缘色
        linestyle='-',      # 实线连接
        linewidth=2,        # 线宽
        color='blue',       # 线条颜色
        label='Data Series')

# 设置标题和标签
ax.set_title("Subject", fontsize=14, fontweight='bold')
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Value", fontsize=12)

# 设置坐标轴范围
ax.set_xlim(-1, 11)
ax.set_ylim(0, 100)

# 设置刻度与网格
# ax.set_xticks(np.arange(0, 11, 1))   # 显示所有整数刻度
# ax.grid(True,
#         linestyle=':',
#         alpha=0.6,
#         which='both')  # 同时显示主次网格

# 添加图例
ax.legend(loc='upper left', framealpha=0.9)

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()