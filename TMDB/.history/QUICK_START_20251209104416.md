# 🚀 快速開始指南

## 📋 專案狀態

**當前最佳模型**: Gradient Boosting (R² = 0.703)
**提交檔案**: `submission_advanced.csv`
**特徵數量**: 83個

---

## ⚡ 5分鐘快速執行

### 1. 基礎版本（無需額外安裝）

```bash
# 執行基礎模型
python simple_main.py

# 輸出: submission.csv
# 性能: R² = 0.688, RMSE = $72.4M
```

### 2. 進階版本（推薦）⭐

```bash
# 執行進階模型
python advanced_model.py

# 輸出: submission_advanced.csv
# 性能: R² = 0.703, RMSE = $75.8M
```

### 3. 查看 EDA 結果

```bash
# 生成 EDA 圖表
python eda_visualization.py

# 輸出: 6張圖表
# - eda_1_revenue_distribution.png
# - eda_2_numerical_features.png
# - eda_3_correlation_matrix.png
# - eda_4_time_trends.png
# - eda_5_genre_analysis.png
# - eda_6_budget_revenue_analysis.png
```

---

## 📦 安裝依賴

### 基礎版本
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 完整版本（推薦）
```bash
pip install -r requirements.txt
# 或
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

---

## 📊 模型性能對比

| 版本 | 檔案 | R² | RMSE | 特徵數 |
|------|------|----|----|--------|
| 基礎 | simple_main.py | 0.688 | $72.4M | 14 |
| 進階 | advanced_model.py | **0.703** | **$75.8M** | **83** |

---

## 🎯 提交到 Kaggle

### 方法 1: 網頁上傳
1. 登入 Kaggle
2. 進入競賽頁面
3. 點擊 "Submit Predictions"
4. 上傳 `submission_advanced.csv`

### 方法 2: Kaggle API
```bash
# 安裝 Kaggle API
pip install kaggle

# 設定 API Token（從 Kaggle 帳戶下載 kaggle.json）
# 放置到 ~/.kaggle/kaggle.json

# 提交
kaggle competitions submit -c tmdb-box-office-prediction -f submission_advanced.csv -m "Advanced model with 83 features"
```

---

## 📁 重要檔案說明

### 模型檔案
- `simple_main.py` - 基礎模型（14特徵，R²=0.688）
- `advanced_model.py` - 進階模型（83特徵，R²=0.703）⭐
- `main.py` - 完整版本（需安裝 XGBoost/LightGBM/CatBoost）

### 分析檔案
- `eda_visualization.py` - 探索性資料分析
- `feature_importance.csv` - 特徵重要性排名

### 文件檔案
- `README.md` - 專案說明
- `SUMMARY.md` - 專案總結 ⭐
- `MODEL_IMPROVEMENTS.md` - 改進記錄
- `NEXT_STEPS.md` - 下一步指南
- `task.md` - 任務規劃
- `report.md` - 詳細報告

### 提交檔案
- `submission.csv` - 基礎模型預測
- `submission_advanced.csv` - 進階模型預測 ⭐

---

## 🔍 檢查結果

### 查看預測統計
```python
import pandas as pd

# 讀取提交檔案
sub = pd.read_csv('submission_advanced.csv')

# 統計資訊
print(f"預測數量: {len(sub)}")
print(f"最小值: ${sub['revenue'].min():,.0f}")
print(f"最大值: ${sub['revenue'].max():,.0f}")
print(f"平均值: ${sub['revenue'].mean():,.0f}")
print(f"中位數: ${sub['revenue'].median():,.0f}")
```

### 查看特徵重要性
```python
import pandas as pd

# 讀取特徵重要性
fi = pd.read_csv('feature_importance.csv')

# 顯示 Top 10
print(fi.head(10))
```

---

## 💡 關鍵特徵

### Top 5 最重要特徵
1. **top_company_mean_revenue** (9.40%) - 製作公司平均票房
2. **top_company_max_revenue** (7.76%) - 製作公司最高票房
3. **budget_x_company_mean** (6.48%) - 預算 × 公司平均票房
4. **budget_x_popularity** (4.93%) - 預算 × 人氣度
5. **budget_x_runtime** (3.57%) - 預算 × 時長

### 關鍵洞察
- 製作公司的歷史數據是最強預測因子
- 交互特徵（budget × 其他特徵）非常有效
- Log 轉換處理偏態分布很重要

---

## 🚀 進一步提升

### 立即可做
```bash
# 1. 安裝進階模型庫
pip install xgboost lightgbm catboost

# 2. 重新訓練
python advanced_model.py

# 預期: R² 提升至 0.75-0.78
```

### 本週目標
- [ ] 超參數調優（使用 RandomizedSearchCV）
- [ ] 5-Fold 交叉驗證
- [ ] 特徵選擇

### 本月目標
- [ ] Stacking 集成
- [ ] 外部數據增強（IMDb 評分）
- [ ] 深度學習模型

詳見 `NEXT_STEPS.md`

---

## 🐛 常見問題

### Q1: 執行時出現 "ModuleNotFoundError"
```bash
# 安裝缺少的套件
pip install [套件名稱]

# 或安裝所有依賴
pip install -r requirements.txt
```

### Q2: 記憶體不足
```bash
# 減少模型參數
# 在 advanced_model.py 中修改:
n_estimators=100  # 從 300 減少到 100
max_depth=15      # 從 25 減少到 15
```

### Q3: 訓練時間太長
```bash
# 使用基礎版本
python simple_main.py

# 或減少特徵數量
# 只使用 Top 20 重要特徵
```

### Q4: 如何查看模型性能？
```python
# 在 advanced_model.py 執行後會自動顯示:
# - Random Forest: RMSE, MAE, R²
# - Gradient Boosting: RMSE, MAE, R²
# - Ensemble: RMSE, MAE, R²
```

---

## 📚 學習資源

### 專案文件
- `SUMMARY.md` - 專案總結（推薦先讀）⭐
- `MODEL_IMPROVEMENTS.md` - 詳細改進記錄
- `NEXT_STEPS.md` - 優化指南
- `report.md` - 完整報告

### 外部資源
- [Kaggle Competition](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

---

## 🎯 成功檢查清單

- [ ] 執行 `advanced_model.py` 成功
- [ ] 生成 `submission_advanced.csv`
- [ ] 查看特徵重要性（Top 10）
- [ ] 提交到 Kaggle
- [ ] 記錄分數和排名
- [ ] 閱讀 `SUMMARY.md`
- [ ] 規劃下一步優化

---

## 📞 需要幫助？

1. 查看 `SUMMARY.md` 了解專案全貌
2. 查看 `MODEL_IMPROVEMENTS.md` 了解改進細節
3. 查看 `NEXT_STEPS.md` 了解優化方向
4. 查看 `report.md` 了解完整報告

---

**祝你在 Kaggle 上取得好成績！** 🏆
