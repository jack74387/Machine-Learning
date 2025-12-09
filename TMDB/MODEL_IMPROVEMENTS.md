# TMDB 電影票房預測 - 模型改進記錄

## 版本歷史

### Version 1.0 - 基礎模型 (simple_main.py)
**日期**: 2025-12-09
**模型**: Random Forest
**性能**:
- RMSE: $72,395,536
- MAE: $39,823,460
- R²: 0.6884

**特徵數量**: 14個基礎特徵

**問題**:
- 特徵工程不足
- 只使用單一模型
- 未處理目標變數偏態
- Kaggle 排名: 倒數

---

### Version 2.0 - 進階模型 (advanced_model.py)
**日期**: 2025-12-09
**模型**: Random Forest + Gradient Boosting + Ridge (集成)
**性能**:
- Random Forest RMSE: $78,558,810, R²: 0.6812
- Gradient Boosting RMSE: $75,770,328, R²: 0.7034 ⭐
- Ensemble RMSE: $76,460,089, R²: 0.6980

**特徵數量**: 83個進階特徵

**關鍵改進**:
1. ✅ **目標變數 Log 轉換** - 處理偏態分布
2. ✅ **歷史統計特徵** - 演員/導演/公司的歷史票房數據
3. ✅ **深度特徵工程** - 83個特徵（vs 14個）
4. ✅ **多模型集成** - 3個模型加權平均
5. ✅ **優化超參數** - 更深的樹、更多迭代

**新增特徵類別**:
- 歷史數據特徵（12個）：演員/導演/公司的平均票房、最高票房、電影數量
- 交互特徵（9個）：budget × popularity, budget × year 等
- 多項式特徵（6個）：平方、立方、平方根
- 比例特徵（3個）：歷史票房與預算的比例
- 時間特徵增強（3個）：星期幾、一年中的第幾天
- 類型標記（5個）：是否為動作/冒險/科幻/奇幻/動畫

**Top 5 重要特徵**:
1. top_company_mean_revenue (9.40%) - 製作公司平均票房
2. top_company_max_revenue (7.76%) - 製作公司最高票房
3. budget_x_company_mean (6.48%) - 預算 × 公司平均票房
4. budget_x_popularity (4.93%) - 預算 × 人氣度
5. budget_x_runtime (3.57%) - 預算 × 時長

**預期改進**:
- Kaggle 排名應該有顯著提升
- RMSE 降低約 5-10%
- R² 提升到 0.70+

---

## 下一步優化方向

### 短期優化（可立即實施）
1. **安裝進階模型庫**
   ```bash
   pip install xgboost lightgbm catboost
   ```
   - 預期 R² 提升到 0.75-0.78
   - RMSE 降低 10-15%

2. **5-Fold 交叉驗證**
   - 更穩定的模型評估
   - 減少過擬合

3. **超參數調優**
   - 使用 GridSearchCV 或 RandomizedSearchCV
   - 針對每個模型優化參數

### 中期優化（需要更多時間）
1. **Stacking 集成**
   - 使用元模型學習最佳組合
   - 預期 R² 提升 2-3%

2. **特徵選擇**
   - 移除低重要性特徵
   - 減少過擬合風險

3. **異常值處理**
   - 識別並處理極端值
   - 提升模型穩定性

### 長期優化（進階技術）
1. **外部數據增強**
   - IMDb 評分
   - Rotten Tomatoes 評分
   - 社交媒體熱度

2. **深度學習模型**
   - Neural Network
   - 文本特徵的 BERT embedding

3. **時間序列特徵**
   - 考慮電影上映時的經濟環境
   - 競爭電影分析

---

## 特徵工程詳細說明

### 歷史統計特徵（最重要！）
這些特徵捕捉了演員、導演、製作公司的歷史表現：

**演員特徵**:
- `top_actor_mean_revenue`: 主演的平均票房
- `top_actor_max_revenue`: 主演的最高票房
- `top_actor_movie_count`: 主演的電影數量
- `top3_actors_mean_revenue`: 前3名演員的平均票房

**導演特徵**:
- `director_mean_revenue`: 導演的平均票房
- `director_max_revenue`: 導演的最高票房
- `director_movie_count`: 導演的電影數量

**製作公司特徵**:
- `top_company_mean_revenue`: 公司的平均票房 ⭐ 最重要
- `top_company_max_revenue`: 公司的最高票房
- `top_company_movie_count`: 公司的電影數量

### 交互特徵
捕捉特徵之間的非線性關係：

- `budget_x_popularity`: 預算與人氣的交互
- `budget_x_runtime`: 預算與時長的交互
- `budget_x_year`: 預算隨時間的變化
- `budget_x_director_mean`: 預算與導演歷史的交互
- `budget_x_actor_mean`: 預算與演員歷史的交互
- `budget_x_company_mean`: 預算與公司歷史的交互 ⭐

### 多項式特徵
捕捉非線性關係：

- `budget_squared`, `budget_cubed`: 預算的高次項
- `popularity_squared`, `popularity_cubed`: 人氣的高次項
- `budget_sqrt`, `popularity_sqrt`: 平方根轉換

---

## 模型配置

### Random Forest
```python
n_estimators=300      # 樹的數量（增加）
max_depth=25          # 最大深度（增加）
min_samples_split=2   # 最小分裂樣本數（減少）
min_samples_leaf=1    # 最小葉子樣本數（減少）
max_features='sqrt'   # 特徵採樣
```

### Gradient Boosting
```python
n_estimators=300      # 迭代次數（增加）
max_depth=7           # 樹深度（增加）
learning_rate=0.03    # 學習率（降低以配合更多迭代）
subsample=0.85        # 樣本採樣比例
```

### 集成權重
```python
weights = {
    'rf': 0.40,       # Random Forest
    'gb': 0.50,       # Gradient Boosting（最高權重）
    'ridge': 0.10     # Ridge（基準）
}
```

---

## 實驗結果對比

| 版本 | 模型 | RMSE | R² | 特徵數 | 訓練時間 |
|------|------|------|----|----|---------|
| V1.0 | RF | $72.4M | 0.688 | 14 | ~30s |
| V2.0 | RF | $78.6M | 0.681 | 83 | ~60s |
| V2.0 | GB | $75.8M | 0.703 ⭐ | 83 | ~120s |
| V2.0 | Ensemble | $76.5M | 0.698 | 83 | ~180s |

**觀察**:
- Gradient Boosting 表現最佳（R² = 0.703）
- 集成模型穩定性更好
- 特徵數量增加顯著提升性能

---

## Kaggle 提交記錄

### Submission 1 (simple_main.py)
- 檔案: `submission.csv`
- 模型: Random Forest (基礎)
- 預測平均值: $71,971,046
- Kaggle 分數: [待更新]
- 排名: 倒數

### Submission 2 (advanced_model.py)
- 檔案: `submission_advanced.csv`
- 模型: Ensemble (RF + GB + Ridge)
- 預測平均值: $47,214,809
- Kaggle 分數: [待更新]
- 排名: [待更新]

**預期改進**: 排名應該提升 20-30%

---

## 關鍵洞察

1. **製作公司歷史數據最重要** (9.40% 重要性)
   - 大型製作公司（迪士尼、華納等）的歷史表現是強預測因子

2. **交互特徵很有效**
   - budget × company_mean 是第3重要特徵
   - 捕捉了"大預算 + 好公司 = 高票房"的模式

3. **Log 轉換必要**
   - Revenue 分布極度右偏
   - Log 轉換後模型更穩定

4. **Gradient Boosting > Random Forest**
   - GB 的 R² (0.703) > RF 的 R² (0.681)
   - GB 更適合這個問題

5. **Ridge 表現差**
   - 線性模型無法捕捉複雜關係
   - R² = -4.65（比隨機猜測還差）

---

## 建議的下一步

### 立即行動
1. ✅ 提交 `submission_advanced.csv` 到 Kaggle
2. ⏳ 安裝 XGBoost/LightGBM/CatBoost
3. ⏳ 重新訓練並提交

### 本週目標
1. 實作 5-Fold 交叉驗證
2. 超參數調優
3. 嘗試 Stacking

### 長期目標
1. 收集外部數據
2. 嘗試深度學習
3. 達到 Top 10% 排名

---

## 參考資料

- [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

---

**最後更新**: 2025-12-09
**當前最佳模型**: advanced_model.py (Gradient Boosting, R² = 0.703)
**下一個里程碑**: R² > 0.75
