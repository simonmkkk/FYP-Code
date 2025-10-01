# 單次隨機接入模擬（對應論文Figure 1和Figure 2的模擬驗證）
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import sys
import os

# 添加上層目錄到路徑以便匯入 settings 和 formulas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.formulas import (
    paper_formula_4_success_approx,
    paper_formula_5_collision_approx
)
from settings import FIGURE_N_VALUES

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

def generate_simulation_vs_approximation_data(n_values=None, n_jobs=1, num_samples=10000):
    """
    生成近似公式 vs 單次接入模擬的比較數據
    用於驗證論文公式(4)和(5)的準確性
    
    Args:
        n_values: 要分析的 N 值列表，None 則從配置讀取
        n_jobs: 並行作業數量
        num_samples: 每個(M,N)點的模擬樣本數
    
    Returns:
        dict: 包含理論值、模擬值和誤差的數據
    """
    # 自動導入配置
    if n_values is None:
        try:
            from settings import FIGURE_N_VALUES
            n_values = FIGURE_N_VALUES
        except ImportError:
            n_values = [14]  # 默認值
    
    results = {}
    
    for N in n_values:
        print(f"\n正在生成 N={N} 的模擬vs近似比較數據...")
        
        # 恢復為每個整數 M（1..10N）皆模擬，與原始版本一致
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
        dict: 誤差數據，格式與single_access_analysis.generate_figure2_data()相同
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