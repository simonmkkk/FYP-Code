# 性能指标计算
import numpy as np
from .formulas import confidence_interval_95

def calculate_performance_metrics(results_array):
    """
    計算平均性能指標
    修改：T_a只考慮有成功設備的樣本（與論文一致）
    使用統一的置信區間計算公式
    """
    # P_S和P_C直接計算所有樣本的平均值
    mean_ps = np.mean(results_array[:, 0])
    mean_pc = np.mean(results_array[:, 2])
    
    # T_a特殊處理：只考慮有效樣本（T_a >= 0，即有成功設備的樣本）
    valid_ta_samples = results_array[results_array[:, 1] >= 0, 1]
    if len(valid_ta_samples) > 0:
        mean_ta = np.mean(valid_ta_samples)
        ci_ta = confidence_interval_95(valid_ta_samples)
    else:
        # 如果沒有任何有效樣本（極端情況）
        mean_ta = 0
        ci_ta = 0
    
    # 計算95%置信區間
    ci_ps = confidence_interval_95(results_array[:, 0])
    ci_pc = confidence_interval_95(results_array[:, 2])
    
    return (mean_ps, mean_ta, mean_pc), (ci_ps, ci_ta, ci_pc)