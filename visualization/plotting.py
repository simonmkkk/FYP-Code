# 繪圖函數
import os
import sys
import matplotlib
# 優先使用可互動後端（顯示視窗），不可用則回退到 Agg
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.formulas import relative_error_percentage

# 設置matplotlib支持中文顯示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

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

def plot_figure1(fig1_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 1: 單次隨機接入中分析模型vs近似公式
    動態處理任意數量的 N 值
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(fig1_data)
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")

    # 若外部提供單一軸，僅在單一 N 場景下於該軸繪製
    if ax is not None:
        if len(available_N_keys) != 1:
            raise ValueError("提供 ax 時，Fig.1 目前僅支援單一 N 值的繪製。請在主程式限制 N_VALUES 為單一值。")
        fig = ax.figure
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
        return fig

    # 否則按原邏輯建立子圖
    num_plots = len(available_N_keys)
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        rows = (num_plots + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if num_plots == 1 else axes
    
    # 動態繪製每個 N 值的子圖
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 支援最多6個子圖
    
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        if i >= len(axes):
            break  # 防止索引超出範圍
            
        N_data = fig1_data[N_key]
        ax = axes[i]
        
        # 成功RAO: 分析模型 vs 近似公式4
        ax.plot(N_data['M_over_N'], N_data['analytical_N_S'], 'b-', 
                linewidth=2, label='Successful RAOs (Analytical, Eq. 3)')
        ax.plot(N_data['M_over_N'], N_data['approx_N_S'], 'b--', 
                linewidth=2, label='Successful RAOs (Approximation, Eq. 4)')
        
        # 碰撞RAO: 分析模型 vs 近似公式5
        ax.plot(N_data['M_over_N'], N_data['analytical_N_C'], 'r-', 
                linewidth=2, label='Collision RAOs (Analytical, Eq. 2)')
        ax.plot(N_data['M_over_N'], N_data['approx_N_C'], 'r--', 
                linewidth=2, label='Collision RAOs (Approximation, Eq. 5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('Normalized RAOs', fontsize=12)
        
        # 動態設置標題
        subplot_label = subplot_labels[i] if i < len(subplot_labels) else f'({chr(97+i)})'
        ax.set_title(f'{subplot_label} N={N_value}: Analytical vs Approximation', 
                    fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 10)
        
        # 動態設置 y 軸範圍
        max_y = max(max(N_data['analytical_N_C']), max(N_data['approx_N_C']))
        ax.set_ylim(0, max_y * 1.1)
    
    # 隱藏多餘的子圖（如果有的話）
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
    
    # 動態設置主標題
    N_values_str = ', '.join(map(str, available_N_values))
    if len(available_N_values) == 1:
        title = f'Fig. 1. Validity range of approximation in single random access (N={N_values_str})'
    else:
        title = f'Fig. 1. Validity range of approximation in single random access (N={N_values_str})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_figure2(fig2_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 2: 絕對近似誤差分析
    動態處理任意數量的 N 值，使用不同的線型和標記區分
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(12, 8))
    
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
    plt.tight_layout()
    return created_fig if created_fig is not None else ax.figure

def plot_single_results(results_array, M, N, I_max):
    """
    绘制单点模拟结果分布图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制接入成功率分布
    axes[0].hist(results_array[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(results_array[:, 0]), color='red', linestyle='dashed', linewidth=2)
    axes[0].set_xlabel('接入成功率 (P_S)')
    axes[0].set_ylabel('频次')
    axes[0].set_title('接入成功率分布')
    axes[0].grid(True, alpha=0.3)
    
    # 绘制平均接入延迟分布
    axes[1].hist(results_array[:, 1], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(results_array[:, 1]), color='red', linestyle='dashed', linewidth=2)
    axes[1].set_xlabel('平均接入延迟 (T_a)')
    axes[1].set_ylabel('频次')
    axes[1].set_title('平均接入延迟分布')
    axes[1].grid(True, alpha=0.3)
    
    # 绘制碰撞概率分布
    axes[2].hist(results_array[:, 2], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(np.mean(results_array[:, 2]), color='red', linestyle='dashed', linewidth=2)
    axes[2].set_xlabel('碰撞概率 (P_C)')
    axes[2].set_ylabel('频次')
    axes[2].set_title('碰撞概率分布')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'模拟结果分布 (M={M}, N={N}, I_max={I_max}, 样本数={len(results_array)})')
    plt.tight_layout()
    plt.show()

def plot_scan_results(param_values, P_S_values, T_a_values, P_C_values, 
                     P_S_theory_values, T_a_theory_values, P_C_theory_values,
                     scan_param, M, I_max, combined=True):
    """
    绘制参数扫描结果（包含模拟和理论值及近似誤差）
    
    Args:
        combined (bool): True - 將Figure 3、4、5合併在同一個窗口中顯示
                        False - 分別顯示三個獨立圖表
    """
    # 計算近似誤差 (相對誤差百分比)
    def calculate_error(simulation, theory):
        """計算相對誤差百分比，使用統一的公式"""
        error = []
        for sim, theo in zip(simulation, theory):
            rel_error = relative_error_percentage(theo, sim)
            error.append(rel_error)
        return error
    
    # 計算各指標的誤差
    P_S_error = calculate_error(P_S_values, P_S_theory_values)
    T_a_error = calculate_error(T_a_values, T_a_theory_values) 
    P_C_error = calculate_error(P_C_values, P_C_theory_values)
    
    if combined:
        # 創建1x3的子圖佈局，一次性顯示Figure 3、4、5
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Figure 3: 接入成功率
        ax1 = axes[0]
        ax1.plot(param_values, P_S_values, 'bo', markersize=6, label='Simulation results')
        ax1.plot(param_values, P_S_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 8)')
        ax1.set_xlabel(f'{scan_param}', fontsize=11)
        ax1.set_ylabel('Access success probability (P_S)', fontsize=11, color='b')
        ax1.set_title(f'Fig. 3. Access success probability vs. {scan_param}\n(M={M}, I_max={I_max})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax1_err = ax1.twinx()
        ax1_err.plot(param_values, P_S_error, 'g--', linewidth=1.5, alpha=0.7, label='Approximation error (%)')
        ax1_err.set_ylabel('Approximation error (%)', fontsize=11, color='g')
        ax1_err.tick_params(axis='y', labelcolor='g')
        ax1_err.legend(loc='upper right', fontsize=9)
        
        # Figure 4: 平均接入延遲
        ax2 = axes[1]
        ax2.plot(param_values, T_a_values, 'bo', markersize=6, label='Simulation results')
        ax2.plot(param_values, T_a_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 9)')
        ax2.set_xlabel(f'{scan_param}', fontsize=11)
        ax2.set_ylabel('Average access delay (T_a)', fontsize=11, color='b')
        ax2.set_title(f'Fig. 4. Average access delay vs. {scan_param}\n(M={M}, I_max={I_max})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(0, 8.0)  # 設置平均接入延遲軸範圍
        ax2.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax2_err = ax2.twinx()
        ax2_err.plot(param_values, T_a_error, 'g--', linewidth=1.5, alpha=0.7, label='Approximation error (%)')
        ax2_err.set_ylabel('Approximation error (%)', fontsize=11, color='g')
        ax2_err.tick_params(axis='y', labelcolor='g')
        ax2_err.legend(loc='lower right', fontsize=9)
        
        # Figure 5: 碰撞概率
        ax3 = axes[2]
        ax3.plot(param_values, P_C_values, 'bo', markersize=6, label='Simulation results')
        ax3.plot(param_values, P_C_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 10)')
        ax3.set_xlabel(f'{scan_param}', fontsize=11)
        ax3.set_ylabel('Collision probability (P_C)', fontsize=11, color='b')
        ax3.set_title(f'Fig. 5. Collision probability vs. {scan_param}\n(M={M}, I_max={I_max})', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_ylim(0, 1.0)
        ax3.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax3_err = ax3.twinx()
        ax3_err.plot(param_values, P_C_error, 'g--', linewidth=1.5, alpha=0.7, label='Approximation error (%)')
        ax3_err.set_ylabel('Approximation error (%)', fontsize=11, color='g')
        ax3_err.set_ylim(0, 2.0)  # 設置近似誤差軸範圍為0-2
        ax3_err.tick_params(axis='y', labelcolor='g')
        ax3_err.legend(loc='lower right', fontsize=9)
        
        # 總標題
        plt.suptitle(f'ALOHA System Performance Analysis: Figures 3-5', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 為總標題留出空間
        plt.show()
        
        return fig
    
    else:
        # 分別顯示三個獨立圖表（原來的方式）
        # Figure 3: 接入成功率
        fig3, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.plot(param_values, P_S_values, 'bo', markersize=8, label='Simulation results')
        ax1.plot(param_values, P_S_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 8)')
        ax1.set_xlabel(f'{scan_param}', fontsize=12)
        ax1.set_ylabel('Access success probability (P_S)', fontsize=12, color='b')
        ax1.set_title(f'Fig. 3. Access success probability vs. {scan_param} (M={M}, I_max={I_max})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax1_err = ax1.twinx()
        ax1_err.plot(param_values, P_S_error, 'g--', linewidth=2, alpha=0.7, label='Approximation error (%)')
        ax1_err.set_ylabel('Approximation error (%)', fontsize=12, color='g')
        ax1_err.tick_params(axis='y', labelcolor='g')
        ax1_err.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Figure 4: 平均接入延遲
        fig4, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(param_values, T_a_values, 'bo', markersize=8, label='Simulation results')
        ax2.plot(param_values, T_a_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 9)')
        ax2.set_xlabel(f'{scan_param}', fontsize=12)
        ax2.set_ylabel('Average access delay (T_a)', fontsize=12, color='b')
        ax2.set_title(f'Fig. 4. Average access delay vs. {scan_param} (M={M}, I_max={I_max})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 10.0)  # 設置平均接入延遲軸範圍為0-10
        ax2.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax2_err = ax2.twinx()
        ax2_err.plot(param_values, T_a_error, 'g--', linewidth=2, alpha=0.7, label='Approximation error (%)')
        ax2_err.set_ylabel('Approximation error (%)', fontsize=12, color='g')
        ax2_err.tick_params(axis='y', labelcolor='g')
        ax2_err.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        # Figure 5: 碰撞概率
        fig5, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        ax3.plot(param_values, P_C_values, 'bo', markersize=8, label='Simulation results')
        ax3.plot(param_values, P_C_theory_values, 'r-', linewidth=2, label='Theoretical results (Eq. 10)')
        ax3.set_xlabel(f'{scan_param}', fontsize=12)
        ax3.set_ylabel('Collision probability (P_C)', fontsize=12, color='b')
        ax3.set_title(f'Fig. 5. Collision probability vs. {scan_param} (M={M}, I_max={I_max})', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_ylim(0, 1.0)
        ax3.tick_params(axis='y', labelcolor='b')
        
        # 添加第二個y軸显示誤差
        ax3_err = ax3.twinx()
        ax3_err.plot(param_values, P_C_error, 'g--', linewidth=2, alpha=0.7, label='Approximation error (%)')
        ax3_err.set_ylabel('Approximation error (%)', fontsize=12, color='g')
        ax3_err.set_ylim(0, 2.0)  # 設置近似誤差軸範圍為0-2
        ax3_err.tick_params(axis='y', labelcolor='g')
        ax3_err.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        return fig3, fig4, fig5

def plot_figure1_simulation_validation(sim_vs_approx_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 1的驗證版本: 近似公式 vs 單次接入模擬
    動態處理任意數量的 N 值
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(sim_vs_approx_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 若外部提供單一軸，僅在單一 N 場景下於該軸繪製
    if ax is not None:
        if len(available_N_keys) != 1:
            raise ValueError("提供 ax 時，Fig.1 目前僅支援單一 N 值的繪製。")
        fig = ax.figure
        N_key = available_N_keys[0]
        N_value = available_N_values[0]
        N_data = sim_vs_approx_data[N_key]
        
        # 成功RAO: 近似公式 vs 模擬結果（使用與 analytical 一致的樣式）
        ax.plot(N_data['M_over_N'], N_data['sim_N_S'], 'ko-', linewidth=1.5, markersize=4, 
                label=f'N={N_value} $N_{{S,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_S'], 'k:', linewidth=1.5, 
                label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
        
        # 碰撞RAO: 近似公式 vs 模擬結果
        ax.plot(N_data['M_over_N'], N_data['sim_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
                label=f'N={N_value} $N_{{C,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_C'], 'k--', linewidth=1.5, 
                label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('RAOs/N', fontsize=12)
        ax.set_title(f'Fig. 1. Simulation and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                    fontsize=11)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        return fig
    
    # 根據可用數據數量設置子圖佈局
    num_plots = len(available_N_keys)
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]  # 轉換為列表以統一處理
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        # 對於3個或更多N值，使用2行布局
        rows = (num_plots + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if num_plots == 1 else axes
    
    # 動態繪製每個 N 值的子圖
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 支援最多6個子圖
    
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        if i >= len(axes):
            break  # 防止索引超出範圍
            
        N_data = sim_vs_approx_data[N_key]
        ax = axes[i]
        
        # 成功RAO: 近似公式 vs 模擬結果（使用與 analytical 一致的樣式）
        ax.plot(N_data['M_over_N'], N_data['sim_N_S'], 'ko-', linewidth=1.5, markersize=4, 
                label=f'N={N_value} $N_{{S,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_S'], 'k:', linewidth=1.5, 
                label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
        
        # 碰撞RAO: 近似公式 vs 模擬結果
        ax.plot(N_data['M_over_N'], N_data['sim_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
                label=f'N={N_value} $N_{{C,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_C'], 'k--', linewidth=1.5, 
                label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('RAOs/N', fontsize=12)
        ax.set_title(f'Fig. 1. Simulation and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                    fontsize=11)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
    
    # 隱藏多餘的子圖（如果有的話）
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_figure2_simulation_validation(error_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 2的驗證版本: 近似公式與模擬的誤差分析
    動態處理任意數量的 N 值，使用不同的線型和標記區分
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(12, 8))
    
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(error_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 動態繪製每個 N 值的誤差曲線（使用與 analytical 一致的樣式）
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        N_data = error_data[N_key]
        
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
    
    # 設置對數縱軸
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
    plt.tight_layout()
    return created_fig if created_fig is not None else ax.figure