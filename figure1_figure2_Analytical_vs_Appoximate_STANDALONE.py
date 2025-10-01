#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成論文Figure 1和Figure 2：單次隨機接入中近似公式的有效範圍分析
完全獨立版本 - 不依賴專案內其他模組

====================================
依賴套件（需要先安裝）：
pip install numpy matplotlib joblib tqdm
====================================

說明：
此文件整合了原本分散在 analysis/single_access_analysis.py、
analysis/formulas.py 和 visualization/plotting.py 中的所有必要程式碼，
可以獨立運行，不需要其他專案模組。
"""

import os
from datetime import datetime
import multiprocessing
from math import factorial, comb
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import matplotlib
# 優先使用可互動後端（顯示視窗），不可用則回退到 Agg
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 設置matplotlib支持中文顯示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# ===== 配置參數 =====
N_VALUES = [3]  # 要分析的 N 值列表，例如 [3, 14] 或 [14]
N_JOBS = 16      # 並行進程數（建議設為 CPU 核心數）
# ===================

# ============================================================================
# 第一部分：數學公式實現（從 formulas.py 複製）
# ============================================================================

def generate_partitions(n: int, k: int, min_val: int = 2):
    """
    生成所有满足条件的整数分割：i1+i2+...+ik = n, 每个ij >= min_val
    使用生成器以減少記憶體使用
    """
    if k == 0:
        if n == 0:
            yield []
        return
    if k == 1:
        if n >= min_val:
            yield [n]
        return
    
    for first in range(min_val, n - min_val * (k - 1) + 1):
        for rest in generate_partitions(n - first, k - 1, min_val):
            yield [first] + rest

def paper_formula_1_pk_probability(M: int, N1: int, k: int) -> float:
    """
    论文公式(1)的完全精确实现：pk(M,N1)
    
    严格按照论文中的多重求和结构实现
    不使用任何概率近似或简化
    """
    if k < 0 or k > min(N1, M // 2):
        return 0.0
    
    total_ways = N1 ** M
    
    # 计算满足条件的方法数
    valid_ways = 0
    
    # 遍历碰撞RAO中的总用户数（从2k到M）
    for total_in_collision in range(2 * k, M + 1):
        # 剩余的用户数分配到非碰撞RAO
        remaining_users = M - total_in_collision
        
        # 非碰撞RAO数量为 N1 - k
        # 每个非碰撞RAO最多1个用户，所以 remaining_users <= N1 - k
        if remaining_users > N1 - k:
            continue
        
        # 生成将total_in_collision个用户分配到k个碰撞RAO的所有方式（每个至少2个用户）
        partitions = generate_partitions(total_in_collision, k, 2)
        
        for partition in partitions:
            # 计算将用户分配到特定分区的方式数
            ways_collision = 1
            remaining = M
            for count in partition:
                ways_collision *= comb(remaining, count)
                remaining -= count
            
            # 计算将剩余用户分配到非碰撞RAO的方式数
            # 从N1-k个非碰撞RAO中选择remaining_users个，每个分配1个用户
            ways_non_collision = comb(N1 - k, remaining_users) * factorial(remaining_users)
            
            # 计算将用户分配到特定RAO的方式数
            ways_specific_assignment = ways_collision * ways_non_collision
            
            # 乘以选择哪k个RAO作为碰撞RAO的方式数
            ways_choose_collision_raos = comb(N1, k)
            
            valid_ways += ways_specific_assignment * ways_choose_collision_raos
    
    pk = valid_ways / total_ways if total_ways > 0 else 0.0
    return pk

def paper_formula_2_collision_raos_exact(M: int, N1: int) -> float:
    """
    论文公式(2)的完全精确实现：NC,1
    
    NC,1 = Σ(k=1 to min{N1,⌊M/2⌋}) k * pk(M,N1)
    """
    if M <= 1 or N1 == 0:
        return 0.0
    
    NC_1 = 0.0
    max_k = min(N1, M // 2)
    
    for k in range(1, max_k + 1):
        pk_val = paper_formula_1_pk_probability(M, N1, k)
        NC_1 += k * pk_val
    
    return NC_1

def paper_formula_3_success_raos_exact(M: int, N1: int) -> float:
    """
    论文公式(3)的完全精确实现：NS,1
    
    使用完整的多重求和结构计算期望的成功用户数
    """
    if M == 0 or N1 == 0:
        return 0.0
    
    NS_1 = 0.0
    max_k = min(N1, M // 2)
    
    # 为每个k值计算概率和条件期望
    for k in range(0, max_k + 1):
        pk_val = paper_formula_1_pk_probability(M, N1, k)
        
        if pk_val == 0:
            continue
        
        # 对于每个k，计算期望的成功用户数
        expected_success_given_k = 0.0
        
        # 遍历碰撞RAO中的总用户数
        for total_in_collision in range(2 * k if k > 0 else 0, M + 1):
            remaining_users = M - total_in_collision
            
            if remaining_users > N1 - k:
                continue
            
            # 计算在给定k和total_in_collision的情况下，这个配置的概率
            ways_this_config = 0
            
            if k == 0:
                # 没有碰撞RAO的特殊情况
                if total_in_collision == 0 and remaining_users <= N1:
                    ways_non_collision = comb(N1, remaining_users) * factorial(remaining_users)
                    ways_this_config = ways_non_collision
            else:
                partitions = generate_partitions(total_in_collision, k, 2)
                for partition in partitions:
                    ways_collision = 1
                    remaining = M
                    for count in partition:
                        ways_collision *= comb(remaining, count)
                        remaining -= count
                    
                    ways_non_collision = comb(N1 - k, remaining_users) * factorial(remaining_users)
                    ways_specific_assignment = ways_collision * ways_non_collision
                    ways_choose_collision_raos = comb(N1, k)
                    
                    ways_this_config += ways_specific_assignment * ways_choose_collision_raos
            
            total_ways = N1 ** M
            prob_this_config = ways_this_config / total_ways if total_ways > 0 else 0
            
            if prob_this_config > 0:
                # 在这个配置下，成功用户数就是remaining_users
                expected_success_given_k += remaining_users * (prob_this_config / pk_val)
        
        NS_1 += expected_success_given_k * pk_val
    
    return NS_1

def paper_formula_4_success_approx(M, N1):
    """
    論文公式(4): NS,1 = M * e^(-M/N1)
    成功RAO近似公式
    """
    return M * np.exp(-M / N1)

def paper_formula_5_collision_approx(M, N1):
    """
    論文公式(5): NC,1 = N1 * (1 - e^(-M/N1) * (1 + M/N1))
    碰撞RAO近似公式
    """
    exp_term = np.exp(-M / N1)
    return N1 * (1 - exp_term * (1 + M/N1))

# ============================================================================
# 第二部分：數據生成函數（從 single_access_analysis.py 複製）
# ============================================================================

def analytical_model(M, N):
    """
    使用論文公式(2)和(3)的分析模型（組合模型）
    """
    N_S = paper_formula_3_success_raos_exact(M, N)
    N_C = paper_formula_2_collision_raos_exact(M, N)
    return N_S, N_C

def approximation_formula(M, N):
    """
    使用論文公式(4)和(5)的近似公式
    """
    N_S = paper_formula_4_success_approx(M, N) 
    N_C = paper_formula_5_collision_approx(M, N)
    return N_S, N_C

def compute_single_point(M, N):
    """
    計算單個(M,N)點的分析模型和近似公式結果
    
    Returns:
        tuple: (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)
    """
    # 分析模型結果
    N_S_anal, N_C_anal = analytical_model(M, N)
    
    # 近似公式結果  
    N_S_approx, N_C_approx = approximation_formula(M, N)
    
    return (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)

def generate_figure1_data(n_values, n_jobs):
    """
    生成Figure 1的數據
    M從1到10N，比較分析模型vs近似公式
    """
    results = {}
    
    for N in n_values:
        print(f"\n正在計算 N={N} 的數據...")
        
        # 目標 M/N 網格（與論文座標一致）
        target_m_over_n = np.arange(0, 10.5, 0.5)  # 0, 0.5, 1, ..., 10
        # 僅保留能被當前 N 精確表示的點（使 M 為整數）
        M_range = []
        for m_n in target_m_over_n:
            m_real = m_n * N
            if abs(round(m_real) - m_real) < 1e-9:
                M_range.append(int(round(m_real)))
        # 移除 0（若存在），並保證至少從 1 開始
        M_range = [m for m in sorted(set(M_range)) if m >= 1]
        print(f"  將計算 {len(M_range)} 個精確 M/N 點: {[f'{m/N:.2f}' for m in M_range]}")
        
        start_time = time.time()
        
        if n_jobs == 1:
            # 單線程計算，顯示詳細進度
            print(f"  單線程逐一計算 {len(M_range)} 個數據點 (M/N = 0 到 10)...")
            
            M_values = []
            analytical_N_S = []
            analytical_N_C = []
            approx_N_S = []
            approx_N_C = []
            
            for idx, M in enumerate(M_range):
                print(f"  計算數據點 {idx+1}: M={M}, M/N={M/N:.2f}", end=' ... ')
                
                M_result, ns_anal_norm, nc_anal_norm, ns_approx_norm, nc_approx_norm = compute_single_point(M, N)
                
                M_values.append(M_result)
                analytical_N_S.append(ns_anal_norm)
                analytical_N_C.append(nc_anal_norm)
                approx_N_S.append(ns_approx_norm)
                approx_N_C.append(nc_approx_norm)
                
                print("完成")
        else:
            # 多核心並行計算
            print(f"  多核心並行計算 {len(M_range)} 個數據點 (使用 {n_jobs} 個核心)...")
            
            # 使用默認的 loky backend，充分利用多核心並行計算，並顯示進度條
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(compute_single_point)(M, N) for M in tqdm(M_range, desc=f"  計算 N={N}", unit="點")
            )
            
            print(f"  ✓ 完成 {len(M_range)} 個數據點的並行計算")
            
            # 解包結果
            M_values = [r[0] for r in results_list]
            analytical_N_S = [r[1] for r in results_list]
            analytical_N_C = [r[2] for r in results_list]
            approx_N_S = [r[3] for r in results_list]
            approx_N_C = [r[4] for r in results_list]
        
        elapsed_time = time.time() - start_time
        print(f"N={N} 計算完成，耗時: {elapsed_time:.2f}秒")
        
        results[f'N_{N}'] = {
            'M_values': M_values,
            'M_over_N': [m/N for m in M_values],  # M/N比值
            'analytical_N_S': analytical_N_S,
            'analytical_N_C': analytical_N_C,
            'approx_N_S': approx_N_S,
            'approx_N_C': approx_N_C
        }
    
    return results

def generate_figure2_data(fig1_data):
    """
    生成Figure 2的數據（優化版本）
    按照論文原文定義計算絕對近似誤差：
    誤差(%) = |分析結果 - 近似結果| / |分析結果| × 100%
    """
    print("重用 Figure 1 數據計算 Figure 2...")
    
    error_results = {}
    
    for key, data in fig1_data.items():
        print(f"正在計算 {key} 的誤差數據...")
        N_S_error = []
        N_C_error = []
        
        for i in range(len(data['analytical_N_S'])):
            # 按照論文定義：|analytical - approximation| / |analytical| * 100%
            anal_ns = data['analytical_N_S'][i] 
            approx_ns = data['approx_N_S'][i]
            error_ns = abs(anal_ns - approx_ns) / abs(anal_ns) * 100
            N_S_error.append(error_ns)
            
            # 計算碰撞RAO的絕對近似誤差
            anal_nc = data['analytical_N_C'][i]
            approx_nc = data['approx_N_C'][i]
            error_nc = abs(anal_nc - approx_nc) / abs(anal_nc) * 100
            N_C_error.append(error_nc)
        
        error_results[key] = {
            'M_over_N': data['M_over_N'],
            'N_S_error': N_S_error,
            'N_C_error': N_C_error
        }
        print(f"  {key} 誤差計算完成")
    
    return error_results

# ============================================================================
# 第三部分：繪圖函數（從 plotting.py 複製並簡化）
# ============================================================================

def extract_n_values_from_data(data_dict):
    """從數據字典中提取 N 值列表"""
    n_keys = []
    n_values = []
    for key in sorted(data_dict.keys()):
        if key.startswith('N_'):
            n_val = int(key.split('_')[1])
            n_keys.append(key)
            n_values.append(n_val)
    return n_keys, n_values

def plot_figure1(fig1_data, ax):
    """
    繪製論文Figure 1: 單次隨機接入中分析模型vs近似公式
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(fig1_data)
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")

    # 單一 N 值的繪製
    if len(available_N_keys) != 1:
        raise ValueError("此獨立版本僅支援單一 N 值的繪製")
    
    N_key = available_N_keys[0]
    N_value = available_N_values[0]
    N_data = fig1_data[N_key]
    
    # 成功RAO: 分析模型 vs 近似公式4
    ax.plot(N_data['M_over_N'], N_data['analytical_N_S'], 'ko-', linewidth=1.5, markersize=4, 
            label=f'N={N_value} $N_{{S,1}}$/N Analytical Model')
    ax.plot(N_data['M_over_N'], N_data['approx_N_S'], 'k:', linewidth=1.5, 
            label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
    
    # 碰撞RAO: 分析模型 vs 近似公式5
    ax.plot(N_data['M_over_N'], N_data['analytical_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
            label=f'N={N_value} $N_{{C,1}}$/N Analytical Model')
    ax.plot(N_data['M_over_N'], N_data['approx_N_C'], 'k--', linewidth=1.5, 
            label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
    
    ax.set_xlabel('M/N', fontsize=12)
    ax.set_ylabel('RAOs/N', fontsize=12)
    ax.set_title(f'Fig. 1. Analytical and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)

def plot_figure2(fig2_data, ax):
    """
    繪製論文Figure 2: 絕對近似誤差分析
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(fig2_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 動態繪製每個 N 值的誤差曲線（使用論文格式）
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        N_data = fig2_data[N_key]
        
        # 根據 N 值選擇不同的樣式
        if i == 0:
            # 第一個 N 值：使用標記點
            # 成功 RAO 誤差（實心圓標記 + 實線）
            ax.plot(N_data['M_over_N'], N_data['N_S_error'], 
                    'ko-', linewidth=1.5, markersize=4,
                    label=f'N={N_value} $N_{{S,1}}$/N')
            # 碰撞 RAO 誤差（空心圓標記 + 實線）
            ax.plot(N_data['M_over_N'], N_data['N_C_error'], 
                    'ko-', fillstyle='none', linewidth=1.5, markersize=4,
                    label=f'N={N_value} $N_{{C,1}}$/N')
        else:
            # 其他 N 值：僅使用線型
            # 成功 RAO 誤差（實線）
            ax.plot(N_data['M_over_N'], N_data['N_S_error'], 
                    'k-', linewidth=1.5,
                    label=f'N={N_value} $N_{{S,1}}$/N')
            # 碰撞 RAO 誤差（虛線）
            ax.plot(N_data['M_over_N'], N_data['N_C_error'], 
                    'k--', linewidth=1.5,
                    label=f'N={N_value} $N_{{C,1}}$/N')
    
    # 設置軸和標籤
    ax.set_xlabel('M/N', fontsize=12)
    ax.set_ylabel('Approximation Error (%)', fontsize=12)
    ax.set_xlim(0, 10)
    
    # 設置對數縱軸 (按照論文Figure 2: 10^-2 到 10^3)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e3)
    
    # 添加網格
    ax.grid(True, alpha=0.3)
    
    # 添加圖例
    ax.legend(fontsize=8, loc='best')
    
    # 動態設置標題
    N_values_str = ' and '.join(map(str, available_N_values))
    title = f'Fig. 2. Absolute approximation error of $N_{{S,1}}$/N and $N_{{C,1}}$/N with N = {N_values_str}'
    
    ax.set_title(title, fontsize=11)

# ============================================================================
# 主程式
# ============================================================================

def main():
    print("=" * 60)
    print("生成論文Figure 1和Figure 2：Analytical vs Approximation")
    print("【完全獨立版本 - 不依賴其他模組】")
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
    combined_path = os.path.join(figures_dir, f"figure1_2_combined_standalone_{timestamp}.png")
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
