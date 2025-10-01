# Figure 1 & Figure 2 獨立腳本說明

## 檔案：figure1_figure2_Analytical_vs_Appoximate_STANDALONE.py

### 作用
這是一個**完全獨立**的 Python 腳本，用於生成論文中的 Figure 1 和 Figure 2，分析單次隨機接入（Single Random Access）中近似公式的有效範圍。

### 與原版的差異
- **原版** (`figure1_figure2_Analytical_vs_Appoximate.py`)：依賴專案內其他模組
  - `analysis/single_access_analysis.py` - 數據生成函數
  - `analysis/formulas.py` - 數學公式實現
  - `visualization/plotting.py` - 繪圖函數
  
- **獨立版** (`figure1_figure2_Analytical_vs_Appoximate_STANDALONE.py`)：
  - ✅ 將所有依賴的代碼整合到單一文件中
  - ✅ 可以單獨運行，不需要其他專案模組
  - ✅ 便於分享或在其他環境中使用

### 安裝依賴

```bash
pip install numpy matplotlib joblib tqdm
```

或使用專案的 requirements.txt：

```bash
pip install -r requirements.txt
```

### 使用方法

1. **配置參數**（在文件頂部）：
   ```python
   N_VALUES = [3]  # 要分析的 N 值列表，例如 [3, 14] 或 [14]
   N_JOBS = 16      # 並行進程數（建議設為 CPU 核心數）
   ```

2. **運行腳本**：
   ```bash
   python figure1_figure2_Analytical_vs_Appoximate_STANDALONE.py
   ```

3. **輸出**：
   - 組合圖表會保存在 `data/figures/` 目錄
   - 文件名格式：`figure1_2_combined_standalone_YYYYMMDD_HHMMSS.png`
   - 終端會顯示計算進度和關鍵結果

### 文件結構

文件內部分為三大部分：

```python
# ============================================================================
# 第一部分：數學公式實現（從 formulas.py 複製）
# ============================================================================
- generate_partitions() - 整數分割生成器
- paper_formula_1_pk_probability() - 論文公式(1)：pk 概率
- paper_formula_2_collision_raos_exact() - 論文公式(2)：碰撞 RAO 數（精確）
- paper_formula_3_success_raos_exact() - 論文公式(3)：成功 RAO 數（精確）
- paper_formula_4_success_approx() - 論文公式(4)：成功 RAO 數（近似）
- paper_formula_5_collision_approx() - 論文公式(5)：碰撞 RAO 數（近似）

# ============================================================================
# 第二部分：數據生成函數（從 single_access_analysis.py 複製）
# ============================================================================
- analytical_model() - 使用精確公式計算
- approximation_formula() - 使用近似公式計算
- compute_single_point() - 計算單點數據
- generate_figure1_data() - 生成 Figure 1 數據（支援多核心並行）
- generate_figure2_data() - 生成 Figure 2 數據（誤差分析）

# ============================================================================
# 第三部分：繪圖函數（從 plotting.py 複製並簡化）
# ============================================================================
- extract_n_values_from_data() - 從數據提取 N 值
- plot_figure1() - 繪製 Figure 1（精確 vs 近似）
- plot_figure2() - 繪製 Figure 2（誤差分析）
```

### 輸出示例

```
============================================================
生成論文Figure 1和Figure 2：Analytical vs Approximation
【完全獨立版本 - 不依賴其他模組】
分析參數：N = [3], 並行進程數 = 16
============================================================

正在生成Figure 1數據（多核心並行計算）...

正在計算 N=3 的數據...
  將計算 21 個精確 M/N 點: ['0.33', '0.67', '1.00', ...]
  多核心並行計算 21 個數據點 (使用 16 個核心)...
  計算 N=3: 100%|██████████| 21/21 [00:01<00:00, 15.23點/s]
  ✓ 完成 21 個數據點的並行計算
N=3 計算完成，耗時: 1.52秒

正在生成Figure 2數據（重用Figure 1數據）...
重用 Figure 1 數據計算 Figure 2...
正在計算 N_3 的誤差數據...
  N_3 誤差計算完成

正在繪製組合圖表...
✓ 組合圖已保存：data\figures\figure1_2_combined_standalone_20251001_120000.png

============================================================
關鍵結果：
  N=3 成功RAO的最大近似誤差: 45.2%

結論：
  1. 近似公式在N較大時更準確
  2. M/N比值影響近似精度
  3. 論文建議實際應用中使用較大的N值
============================================================
```

### 優勢

1. **完全獨立**：不需要專案的其他文件，可以單獨分享或使用
2. **易於理解**：所有相關代碼都在一個文件中，便於閱讀和修改
3. **高效計算**：支援多核心並行計算，大幅提升速度
4. **便於部署**：只需安裝 4 個 Python 套件即可運行

### 注意事項

- 計算複雜度較高，建議使用較小的 N 值進行測試（如 N=3）
- 多核心並行計算需要足夠的記憶體
- 對於大型 N 值（如 N=100），計算時間可能較長

### 適用場景

- 需要在其他環境中重現論文結果
- 不想安裝整個專案，只需要 Figure 1 和 Figure 2
- 教學或演示用途
- 需要修改或擴展分析功能
