# 配置参数设置
import numpy as np

# 运行模式选择
RUN_MODE = 'scan'  # 'single': 单点模拟, 'scan': 参数扫描

# 通用参数
M = 100           # 设备总数
I_max = 10        # 最大接入周期数（最大重传次数）
NUM_WORKERS = 16   # 并行计算的进程数（建议设置为CPU核心数）

# 樣本數設定（論文使用10^7 = 10,000,000）
NUM_SAMPLES = 5000  # 快速測試用較少樣本

# 單點模擬參數
N = 40           # 每个接入周期(AC)的随机接入机会(RAO)数量

# 参数扫描设置 (仅在RUN_MODE='scan'时生效)
SCAN_PARAM = 'N'   # 扫描参数: 'N', 'M', 'I_max'
SCAN_RANGE = range(5, 46, 1)  # 快速測試：N=20,25,30,35,40,45

# Figure 1 & 2 的 N 值配置
FIGURE_N_VALUES = [3]  # 修改這個列表即可改變要分析的 N 值

# Figure 1 & 2 的並行計算設置
FIGURE_N_JOBS = 16      # Figure分析的並行進程數（建議設置為CPU核心數）

# 输出设置
PLOT_RESULTS = True       # 是否绘制结果图
SAVE_TO_CSV = True        # 是否保存结果到CSV文件