# 理论计算方法（基于论文）
import numpy as np
from .formulas import (
    paper_formula_6_success_per_cycle,
    paper_formula_7_next_contending_devices, 
    paper_formula_8_access_success_probability,
    paper_formula_9_mean_access_delay,
    paper_formula_10_collision_probability,
    paper_formula_5_collision_approx
)

def theoretical_calculation(M, N, I_max):
    """
    使用论文中的理论方法计算性能指标
    严格按照论文公式实现，調用統一的公式模塊
    
    Args:
        M: 設備總數
        N: 每個AC的RAO數
        I_max: 最大AC數
    
    Returns:
        tuple: (P_S, T_a, P_C, N_s_list, K_list)
    """
    K = [M]  # K[0] = M，但實際從K[1]開始使用
    N_s = []  # N_s[i-1]對應第i個AC的成功設備數
    N_c = []  # N_c[i-1]對應第i個AC的碰撞RAO數
    
    # 按論文公式計算每個AC
    for i in range(1, I_max + 1):  # i從1到I_max
        if len(K) <= i-1 or K[i-1] <= 0:
            N_s.append(0)
            N_c.append(0)
            K.append(0)
            continue
        
        # 論文公式(6): N_{S,i} ≈ K_i * exp(-K_i/N)
        current_K = K[i-1]
        N_s_i = paper_formula_6_success_per_cycle(current_K, N)
        N_s.append(N_s_i)
        
        # 論文公式(5): N_{C,i} ≈ N * (1 - exp(-K_i/N) * (1 + K_i/N))
        N_c_i = paper_formula_5_collision_approx(current_K, N)
        N_c.append(N_c_i)
        
        # 論文公式(7): K_{i+1} = K_i(1 - e^(-K_i/N)) 
        # 這等於 K_i - N_{S,i}，但我們使用論文的完整公式
        K_i_plus_1 = paper_formula_7_next_contending_devices(current_K, N)
        K.append(K_i_plus_1)
    
    # 計算性能指標
    # 論文公式(8): P_S = sum(N_{S,i}) / M
    P_S = paper_formula_8_access_success_probability(N_s, M)
    
    # 論文公式(9): T_a = sum(i * N_{S,i}) / sum(N_{S,i})
    T_a = paper_formula_9_mean_access_delay(N_s)
    
    # 論文公式(10): P_C = sum(N_{C,i}) / sum(N_i)
    P_C = paper_formula_10_collision_probability(N_c, I_max, N)
    
    return P_S, T_a, P_C, N_s, K