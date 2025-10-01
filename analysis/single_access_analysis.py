# 單次隨機接入分析（對應論文Figure 1和Figure 2）
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
# 導入論文的完全精確公式
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加上層目錄到路徑以便匯入 settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .formulas import (
    paper_formula_1_pk_probability,
    paper_formula_2_collision_raos_exact, 
    paper_formula_3_success_raos_exact,
    paper_formula_4_success_approx,
    paper_formula_5_collision_approx
)
from settings import FIGURE_N_VALUES

def compute_single_point(M, N):
    """
    計算單個(M,N)點的分析模型和近似公式結果
    
    Args:
        M: 設備總數
        N: RAO總數
    
    Returns:
        tuple: (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)
    """
    # 分析模型結果
    N_S_anal, N_C_anal = analytical_model(M, N)
    
    # 近似公式結果  
    N_S_approx, N_C_approx = approximation_formula(M, N)
    
    return (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)

def analytical_model(M, N):
    """
    使用論文公式(2)和(3)的分析模型（組合模型）
    調用統一公式模塊的精確計算
    
    Args:
        M: 設備總數
        N: RAO總數
    
    Returns:
        tuple: (N_S, N_C) 成功RAO數和碰撞RAO數
    """
    # 使用論文完全精確公式(2)和(3)計算
    N_S = paper_formula_3_success_raos_exact(M, N)
    N_C = paper_formula_2_collision_raos_exact(M, N)
    
    return N_S, N_C

def approximation_formula(M, N):
    """
    使用論文公式(4)和(5)的近似公式
    調用統一公式模塊的近似計算
    
    Args:
        M: 設備總數
        N: RAO總數
    
    Returns:
        tuple: (N_S, N_C) 成功RAO數和碰撞RAO數
    """
    # 使用論文近似公式(4)和(5)計算
    N_S = paper_formula_4_success_approx(M, N) 
    N_C = paper_formula_5_collision_approx(M, N)
    
    return N_S, N_C

def generate_figure1_data(n_values=None, n_jobs=None):
    """
    生成Figure 1的數據
    M從1到10N，比較分析模型vs近似公式
    
    Args:
        n_values: 要分析的 N 值列表，None 則從配置讀取
        n_jobs: 並行作業數量
                - None: 自動從配置檔案讀取 FIGURE_N_JOBS
                - 1: 單線程順序計算
                - >1: 多核心並行計算
    """
    # 自動導入配置
    if n_values is None:
        try:
            from settings import FIGURE_N_VALUES
            n_values = FIGURE_N_VALUES
        except ImportError:
            n_values = [14]  # 默認值
    
    if n_jobs is None:
        try:
            from settings import FIGURE_N_JOBS
            n_jobs = FIGURE_N_JOBS
        except ImportError:
            n_jobs = 1  # 默認單線程
    
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

def generate_figure2_data(fig1_data=None):
    """
    生成Figure 2的數據（優化版本）
    按照論文原文定義計算絕對近似誤差：
    "the absolute difference of the analytical results and approximation results and normalized by the analytical results"  
    誤差(%) = |分析結果 - 近似結果| / |分析結果| × 100%
    
    Args:
        fig1_data: 可選的 figure1 數據，如果提供則重用，否則重新計算
    """
    if fig1_data is None:
        print("重新計算 Figure 1 數據用於 Figure 2...")
        fig1_data = generate_figure1_data()
    else:
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