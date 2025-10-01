### Fig.1/2 Analytical 執行流程與依賴套件說明

本文檔說明當您執行 `aloha_simulation/figure1_figure2_Analytical_vs_Appoximate.py` 時，會依序運行哪些程式碼、主要數學操作與所需套件。

---

## 一、入口腳本與高階流程

檔案：`aloha_simulation/figure1_figure2_Analytical_vs_Appoximate.py`

步驟：
1. 載入模組與時間戳設置
2. 呼叫 `analysis.single_access_analysis.generate_figure1_data()` 生成 Fig.1 的 Analytical vs Approximation 數據
3. 呼叫 `analysis.single_access_analysis.generate_figure2_data(fig1_data)` 基於 Fig.1 數據生成 Fig.2 誤差
4. 繪圖：
   - `visualization.plotting.plot_figure1(fig1_data)`
   - `visualization.plotting.plot_figure2(fig2_data)`
5. 以時間戳檔名輸出圖片至 `data/figures/`
6. 列印關鍵觀察與結論

配置來源：`aloha_simulation/settings.py`
- `FIGURE_N_VALUES`：要分析的 N 列表
- `FIGURE_N_JOBS`：並行進程數

---

## 二、Fig.1 數據生成（Analytical 與 Approximation）

檔案：`aloha_simulation/analysis/single_access_analysis.py`

核心函數：
- `generate_figure1_data(n_jobs=None, use_optimized=True)`
  - 讀取 `FIGURE_N_VALUES`
  - 對每個 N，建立 `M_range = 1..10N`
  - 逐點計算 `(M,N)`：
    - Analytical（精確）：`paper_formula_3_success_raos_exact`、`paper_formula_2_collision_raos_exact`
    - Approximation：`paper_formula_4_success_approx`、`paper_formula_5_collision_approx`
  - 並行：使用 `joblib.Parallel` 逐點（每個 M 一個任務），後端為預設 `loky`（多進程）
  - 若 `use_optimized=True` 且 `n_jobs>1`，改走優化路徑（見下一節）

數學重點：
- Analytical：組合計數與期望加總（無近似），遍歷碰撞配置並按機率加權
- Approximation：以 `e^{-M/N}` 為核心的封閉式近似

---

## 三、優化逐點並行與進度（不分批）

檔案：`aloha_simulation/analysis/vectorized_calculation.py`

核心函數：
- `compute_figure1_data_optimized(N, n_jobs=16, batch_size=10)`
  - 逐點並行：對 `M=1..10N`，每個 M 為獨立任務
  - 計算內容同上（Analytical 精確 + Approximation），輸出皆做 `÷N` 正規化
  - 進度顯示：
    - `tqdm` 總進度條（總任務數 = 10N）
    - 逐點列印 `[開始] M=...`、`[完成] M=...`
  - 並行框架：`joblib.Parallel(n_jobs=FIGURE_N_JOBS)` 預設後端 `loky`

---

## 四、Fig.2 誤差數據生成

檔案：`aloha_simulation/analysis/single_access_analysis.py`

核心函數：
- `generate_figure2_data(fig1_data)`
  - 公式：`誤差(%) = |Analytical - Approximation| / |Analytical| × 100%`
  - 對成功 RAO 與碰撞 RAO 各自計算誤差序列

---

## 五、繪圖與輸出

檔案：`aloha_simulation/visualization/plotting.py`

函數與輸出：
- `plot_figure1(fig1_data)`：繪製 Fig.1（Analytical vs Approximation）
- `plot_figure2(fig2_data)`：繪製 Fig.2（誤差對數縱軸）
- 圖片輸出目錄：`data/figures/`

後端選擇：動態設定 matplotlib backend（`TkAgg`/`Agg`），以支持互動或多進程環境。

---

## 六、所需套件清單（Analytical 流程）

計算與數學：
- numpy（`np.exp`、統計/陣列）
- math（`factorial`、`comb`）
- itertools（迭代輔助）
- functools（`lru_cache`）

並行與進度：
- joblib（`Parallel`、`delayed`，預設 `loky` 多進程）
- tqdm（進度條及與 joblib 整合）

繪圖：
- matplotlib（`pyplot`、中文字體設定）

其他：
- time（耗時統計列印）
- os、sys（路徑與環境）
- multiprocessing（Windows 兼容啟動）

安裝參考（若需要）：
```bash
pip install numpy joblib tqdm matplotlib
```

---

## 七、輸出檔名格式

- Fig.1：`data/figures/figure1_single_access_YYYYMMDD_HHMMSS.png`
- Fig.2：`data/figures/figure2_approximation_error_YYYYMMDD_HHMMSS.png`

---

## 八、可調參數（於 `aloha_simulation/settings.py`）

- `FIGURE_N_VALUES = [ ... ]`：指定要分析的 N 值清單（例如 `[14]`）
- `FIGURE_N_JOBS = 16`：並行進程數（建議=CPU核心數）



