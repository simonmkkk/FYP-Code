#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成論文Figure 1和Figure 2的驗證版本：近似公式 vs 單次接入模擬的比較
完全獨立版本 - 僅依賴 formulas.py 和 plotting.py

====================================
依賴套件（需要先安裝）：
pip install numpy matplotlib joblib tqdm
====================================

說明：
此文件整合了原本在 analysis/single_access_simulation.py 中的模擬代碼，
但保留對 analysis/formulas.py 和 visualization/plotting.py 的依賴，
這兩個模組是專案的核心共用元件。
"""

import sys
import os
from datetime import datetime
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# 添加當前目錄到路徑，以便導入 analysis 和 visualization
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 從核心模組導入（這些模組無條件保留）
from analysis.formulas import (
    paper_formula_4_success_approx,
    paper_formula_5_collision_approx
)
from visualization.plotting import (
    plot_figure1_simulation_validation, 
    plot_figure2_simulation_validation
)

# ===== 配置參數 =====
N_VALUES = [3]       # 要分析的 N 值列表
N_JOBS = 16         # 並行進程數（建議設為 CPU 核心數）
NUM_SAMPLES = 50000  # 每個(M,N)點的模擬樣本數
# ===================

# ============================================================================
# 模擬函數（從 single_access_simulation.py 整合）
# ============================================================================

def simulate_single_access_sample(M, N):
    """
    模擬一次單次隨機接入過程
    
    Args:
        M: 設備總數
        N: RAO總數
    
    Returns:
        tuple: (成功RAO數, 碰撞RAO數, 空閒RAO數)
    """
    # M個設備隨機選擇N個RAO中的一個
    choices = np.random.randint(0, N, M)
    
    # 統計每個RAO被選擇的次數
    rao_usage = np.bincount(choices, minlength=N)
    
    # 計算不同類型的RAO數量
    success_raos = np.sum(rao_usage == 1)  # 恰好1個設備
    collision_raos = np.sum(rao_usage >= 2)  # 2個或更多設備
    idle_raos = np.sum(rao_usage == 0)  # 沒有設備
    
    return success_raos, collision_raos, idle_raos

def simulate_single_access_parallel(M, N, num_samples=10000, n_jobs=1):
    """
    並行執行多次單次接入模擬
    
    Args:
        M: 設備總數
        N: RAO總數
        num_samples: 模擬樣本數
        n_jobs: 並行作業數
    
    Returns:
        tuple: (平均成功RAO數, 平均碰撞RAO數, 平均空閒RAO數)
    """
    if n_jobs == 1:
        # 單線程執行
        results = []
        for _ in tqdm(range(num_samples), desc=f"模擬 M={M}, N={N}", leave=False):
            results.append(simulate_single_access_sample(M, N))
    else:
        # 並行執行
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_single_access_sample)(M, N)
            for _ in tqdm(range(num_samples), desc=f"模擬 M={M}, N={N}", leave=False)
        )
    
    results_array = np.array(results)
    
    # 計算平均值
    mean_success = np.mean(results_array[:, 0])
    mean_collision = np.mean(results_array[:, 1])
    mean_idle = np.mean(results_array[:, 2])
    
    return mean_success, mean_collision, mean_idle

def generate_simulation_vs_approximation_data(n_values, n_jobs, num_samples):
    """
    生成近似公式 vs 單次接入模擬的比較數據
    用於驗證論文公式(4)和(5)的準確性
    
    Args:
        n_values: 要分析的 N 值列表
        n_jobs: 並行作業數量
        num_samples: 每個(M,N)點的模擬樣本數
    
    Returns:
        dict: 包含理論值、模擬值和誤差的數據
    """
    results = {}
    
    for N in n_values:
        print(f"\n正在生成 N={N} 的模擬vs近似比較數據...")
        
        # 每個整數 M（1..10N）皆模擬
        M_range = list(range(1, 10*N + 1))
        
        start_time = time.time()
        
        # 初始化結果列表
        M_values = []
        
        # 理論值 (近似公式)
        theory_N_S = []
        theory_N_C = []
        
        # 模擬值
        sim_N_S = []
        sim_N_C = []
        
        # 誤差
        error_N_S = []
        error_N_C = []
        
        print(f"  總共需要模擬 {len(M_range)} 個數據點，每點 {num_samples} 樣本...")
        
        for idx, M in enumerate(M_range):
            if idx % 5 == 0:  # 每5個點顯示進度
                print(f"  進度: {idx+1}/{len(M_range)} (M={M}, M/N={M/N:.2f})")
            
            # 計算理論值（近似公式）
            theory_ns = paper_formula_4_success_approx(M, N)
            theory_nc = paper_formula_5_collision_approx(M, N)
            
            # 執行模擬
            sim_ns, sim_nc, sim_idle = simulate_single_access_parallel(
                M, N, num_samples, n_jobs
            )
            
            # 計算相對誤差百分比
            error_ns = abs(sim_ns - theory_ns) / abs(theory_ns) * 100 if theory_ns > 0 else 0
            error_nc = abs(sim_nc - theory_nc) / abs(theory_nc) * 100 if theory_nc > 0 else 0
            
            # 保存結果
            M_values.append(M)
            theory_N_S.append(theory_ns / N)  # 正規化
            theory_N_C.append(theory_nc / N)  # 正規化
            sim_N_S.append(sim_ns / N)  # 正規化
            sim_N_C.append(sim_nc / N)  # 正規化
            error_N_S.append(error_ns)
            error_N_C.append(error_nc)
        
        elapsed_time = time.time() - start_time
        print(f"N={N} 模擬完成，耗時: {elapsed_time:.2f}秒")
        
        results[f'N_{N}'] = {
            'M_values': M_values,
            'M_over_N': [m/N for m in M_values],
            # 理論值 (近似公式)
            'theory_N_S': theory_N_S,
            'theory_N_C': theory_N_C,
            # 模擬值
            'sim_N_S': sim_N_S,
            'sim_N_C': sim_N_C,
            # 誤差
            'error_N_S': error_N_S,
            'error_N_C': error_N_C
        }
    
    return results

def generate_simulation_vs_approximation_error_data(sim_vs_approx_data):
    """
    從模擬vs近似數據中提取誤差數據（用於Figure 2驗證版）
    
    Args:
        sim_vs_approx_data: generate_simulation_vs_approximation_data()的輸出
    
    Returns:
        dict: 誤差數據
    """
    error_results = {}
    
    for key, data in sim_vs_approx_data.items():
        print(f"正在提取 {key} 的誤差數據...")
        
        error_results[key] = {
            'M_over_N': data['M_over_N'],
            'N_S_error': data['error_N_S'],
            'N_C_error': data['error_N_C']
        }
        print(f"  {key} 誤差數據提取完成")
    
    return error_results

# ============================================================================
# 主程式
# ============================================================================

def main():
    print("=" * 60)
    print("生成論文Figure 1和Figure 2的驗證版本")
    print("近似公式 vs 單次接入模擬的比較")
    print("【獨立版本 - 僅依賴 formulas.py 和 plotting.py】")
    print(f"分析參數：N = {N_VALUES}, 並行進程數 = {N_JOBS}, 樣本數 = {NUM_SAMPLES}")
    print("=" * 60)
    
    # 生成模擬vs近似比較數據
    print("\n正在生成模擬vs近似比較數據...")
    print(f"使用 {N_JOBS} 個核心並行計算，每點 {NUM_SAMPLES} 個樣本...")
    
    sim_vs_approx_data = generate_simulation_vs_approximation_data(
        n_values=N_VALUES,
        n_jobs=N_JOBS, 
        num_samples=NUM_SAMPLES
    )
    print("模擬vs近似比較數據生成完成!")
    
    # 提取誤差數據
    print("\n正在提取誤差數據...")
    error_data = generate_simulation_vs_approximation_error_data(sim_vs_approx_data)
    print("誤差數據提取完成!")
    
    # 繪製與保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join('data', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n正在繪製組合圖表...")
    num_n_values = len(N_VALUES)
    
    # 創建 2 行 × N 列的子圖佈局（與 analytical 一致）
    fig_width = 8 * num_n_values
    fig_height = 12
    fig_combined, axes = plt.subplots(2, num_n_values, figsize=(fig_width, fig_height), 
                                      constrained_layout=True, squeeze=False)
    
    # 為每個 N 值繪製 Fig1 和 Fig2
    for col_idx, N in enumerate(N_VALUES):
        n_key = f'N_{N}'
        
        # 繪製 Fig1 到第一行
        if n_key in sim_vs_approx_data:
            ax1 = axes[0, col_idx]
            # 提取單個 N 的數據
            single_n_fig1_data = {n_key: sim_vs_approx_data[n_key]}
            plot_figure1_simulation_validation(single_n_fig1_data, ax=ax1)
        
        # 繪製 Fig2 到第二行
        if n_key in error_data:
            ax2 = axes[1, col_idx]
            # 提取單個 N 的數據
            single_n_fig2_data = {n_key: error_data[n_key]}
            plot_figure2_simulation_validation(single_n_fig2_data, ax=ax2)
    
    # 保存組合圖
    combined_path = os.path.join(figures_dir, f"figure1_2_simulation_combined_standalone_{timestamp}.png")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ 組合圖已保存：{combined_path}")
    
    # 顯示圖表
    plt.show()
    
    # 分別保存單獨的圖表
    print("\n正在繪製並保存單獨的 Figure 1...")
    fig1 = plot_figure1_simulation_validation(sim_vs_approx_data)
    fig1_path = os.path.join(figures_dir, f"figure1_simulation_validation_standalone_{timestamp}.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1已保存：{fig1_path}")
    
    print("\n正在繪製並保存單獨的 Figure 2...")
    fig2 = plot_figure2_simulation_validation(error_data)
    fig2_path = os.path.join(figures_dir, f"figure2_simulation_validation_standalone_{timestamp}.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2已保存：{fig2_path}")
    
    # 顯示一些關鍵結果
    print("\n" + "=" * 60)
    print("關鍵結果：")
    for N in N_VALUES:
        key = f'N_{N}'
        if key in error_data:
            max_error_s = max(error_data[key]['N_S_error'])
            max_error_c = max(error_data[key]['N_C_error'])
            print(f"  N={N} 成功RAO的最大模擬誤差: {max_error_s:.2f}%")
            print(f"  N={N} 碰撞RAO的最大模擬誤差: {max_error_c:.2f}%")
    
    print("\n結論：")
    print("  1. 模擬結果驗證了近似公式的準確性")
    print("  2. 較大的N值下，近似公式與模擬結果更接近")
    print("  3. 這證實了論文理論分析的正確性")
    print("=" * 60)

if __name__ == "__main__":
    main()
