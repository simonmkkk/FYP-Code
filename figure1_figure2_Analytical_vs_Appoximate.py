#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成論文Figure 1和Figure 2：單次隨機接入中近似公式的有效範圍分析"""

import sys
import os
from datetime import datetime
import multiprocessing
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.single_access_analysis import generate_figure1_data, generate_figure2_data
from visualization.plotting import plot_figure1, plot_figure2

# ===== 配置參數 =====
N_VALUES = [3]  # 要分析的 N 值列表，例如 [3, 14] 或 [14]
N_JOBS = 16      # 並行進程數（建議設為 CPU 核心數）
# ===================

def main():
    print("=" * 60)
    print("生成論文Figure 1和Figure 2：Analytical vs Approximation")
    print(f"分析參數：N = {N_VALUES}, 並行進程數 = {N_JOBS}")
    print("=" * 60)
    
    # 生成數據
    print("\n正在生成Figure 1數據（多核心並行計算）...")
    fig1_data = generate_figure1_data(n_values=N_VALUES, n_jobs=N_JOBS)
    
    print("\n正在生成Figure 2數據（重用Figure 1數據）...")
    fig2_data = generate_figure2_data(fig1_data)
    
    # 繪製與保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join('data', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n正在繪製組合圖表...")
    num_n_values = len(N_VALUES)
    
    # 創建 2 行 × N 列的子圖佈局
    # 第一行：所有 N 值的 Fig1
    # 第二行：所有 N 值的 Fig2
    fig_width = 8 * num_n_values  # 每個子圖寬度 8
    fig_height = 12  # 總高度 12（每行 6）
    fig_combined, axes = plt.subplots(2, num_n_values, figsize=(fig_width, fig_height), 
                                      constrained_layout=True, squeeze=False)
    
    # 為每個 N 值繪製 Fig1 和 Fig2
    for col_idx, N in enumerate(N_VALUES):
        n_key = f'N_{N}'
        
        # 繪製 Fig1 到第一行
        if n_key in fig1_data:
            ax1 = axes[0, col_idx]
            # 提取單個 N 的數據
            single_n_fig1_data = {n_key: fig1_data[n_key]}
            plot_figure1(single_n_fig1_data, ax=ax1)
        
        # 繪製 Fig2 到第二行
        if n_key in fig2_data:
            ax2 = axes[1, col_idx]
            # 提取單個 N 的數據
            single_n_fig2_data = {n_key: fig2_data[n_key]}
            plot_figure2(single_n_fig2_data, ax=ax2)
    
    # 保存組合圖
    combined_path = os.path.join(figures_dir, f"figure1_2_combined_{timestamp}.png")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ 組合圖已保存：{combined_path}")
    
    # 顯示圖表
    plt.show()
    
    # 顯示關鍵結果
    print("\n" + "=" * 60)
    print("關鍵結果：")
    for N in N_VALUES:
        if f'N_{N}' in fig2_data:
            max_error = max(fig2_data[f'N_{N}']['N_S_error'])
            print(f"  N={N} 成功RAO的最大近似誤差: {max_error:.1f}%")
    
    print("\n結論：")
    print("  1. 近似公式在N較大時更準確")
    print("  2. M/N比值影響近似精度")
    print("  3. 論文建議實際應用中使用較大的N值")
    print("=" * 60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()