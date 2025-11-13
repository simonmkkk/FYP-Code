# 文件读写功能
import csv
import os
from datetime import datetime
import numpy as np

def save_single_results_to_csv(results_array, M, N, I_max, num_samples, output_dir='data_graph/results'):
    """
    保存单点模拟结果到CSV文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_M{M}_N{N}_Imax{I_max}_samples{num_samples}_{timestamp}.csv"
    
    # 确保结果目录存在
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入文件头
        writer.writerow(['参数配置', f'M={M}, N={N}, I_max={I_max}, 样本数={num_samples}'])
        writer.writerow([])
        writer.writerow(['样本索引', '接入成功率(P_S)', '平均接入延迟(T_a)', '碰撞概率(P_C)'])
        
        # 写入数据
        for i, (ps, ta, pc) in enumerate(results_array):
            writer.writerow([i+1, ps, ta, pc])
        
        # 写入统计信息
        mean_ps, mean_ta, mean_pc = np.mean(results_array, axis=0)
        std_ps, std_ta, std_pc = np.std(results_array, axis=0)
        
        writer.writerow([])
        writer.writerow(['统计量', '平均值', '标准差'])
        writer.writerow(['接入成功率(P_S)', mean_ps, std_ps])
        writer.writerow(['平均接入延迟(T_a)', mean_ta, std_ta])
        writer.writerow(['碰撞概率(P_C)', mean_pc, std_pc])
    
    print(f"结果已保存到: {filepath}")
    return filepath

def save_scan_results_to_csv(param_values, P_S_values, T_a_values, P_C_values, scan_param, M, I_max, output_dir='data_graph/results'):
    """
    保存扫描结果到CSV文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_results_{scan_param}_M{M}_Imax{I_max}_{timestamp}.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入文件头
        writer.writerow([f'{scan_param}值扫描结果'])
        writer.writerow([f'参数: M={M}, I_max={I_max}'])
        writer.writerow([])
        writer.writerow([scan_param, '接入成功率(P_S)', '平均接入延迟(T_a)', '碰撞概率(P_C)'])
        
        # 写入数据
        for i in range(len(param_values)):
            writer.writerow([param_values[i], P_S_values[i], T_a_values[i], P_C_values[i]])
    
    print(f"数据已保存到: {filepath}")
    return filepath