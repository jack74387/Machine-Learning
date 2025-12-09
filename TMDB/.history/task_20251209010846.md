# TMDB 電影票房預測專案 - Taskmaster 任務規劃

## 專案概述
使用 TMDB (The Movie Database) 電影資料集預測電影票房收入

**資料來源**: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

**目標**: 預測電影的 revenue (票房收入)

---

## 任務分解 (Taskmaster Method)

### Phase 1: 問題定義與資料理解 ✅
- [x] 確認問題類型：回歸問題 (預測連續值 revenue)
- [x] 載入並檢視資料集
- [x] 分析資料欄位與特徵
- [x] 檢查資料品質（缺失值、異常值）

### Phase 2: 探索性資料分析 (EDA) ✅
- [x] 分析目標變數 revenue 的分布
- [x] 分析數值型特徵（budget, popularity, runtime）
- [x] 分析類別型特徵（genres, cast, crew, keywords）
- [x] 特徵相關性分析
- [x] 視覺化關鍵發現（已完成，生成 6 張圖表）

### Phase 3: 資料預處理 ✅
- [x] 處理缺失值
- [x] 處理 JSON 格式欄位（genres, cast, crew, keywords 等）
- [x] 特徵工程
  - [x] 從 release_date 提取時間特徵
  - [x] 從 cast/crew 提取關鍵人物
  - [x] 從 genres 提取類型特徵
  - [x] 創建衍生特徵（如 budget_popularity_ratio）
- [x] 特徵編碼與標準化
- [x] 分割訓練集與驗證集

### Phase 4: 模型建立與訓練 ✅
- [x] Baseline 模型（Random Forest）
- [x] 進階模型
  - [x] Random Forest ✅
  - [x] XGBoost（已實作，Kaggle 版本支援）
  - [x] LightGBM（已實作，Kaggle 版本支援）
  - [x] CatBoost（已實作，Kaggle 版本支援）
- [x] 超參數調優（基礎版本）
- [x] 交叉驗證

### Phase 5: 模型融合 (Voting/Ensemble) ✅
- [x] 實作 Voting Regressor
- [x] 實作多模型融合策略
- [x] 比較不同融合策略
- [ ] 實作 Stacking（進階，可選）
- [ ] 實作 Blending（進階，可選）

### Phase 6: 評估與優化 ✅
- [x] 評估指標：RMSE, MAE, R²
- [x] 特徵重要性分析
- [x] 交叉驗證評估
- [x] 模型性能比較

### Phase 7: 預測與提交 ✅
- [x] 對測試集進行預測
- [x] 生成提交檔案
- [x] 結果分析

---

## 評估指標
- **主要指標**: RMSE (Root Mean Squared Error)
- **次要指標**: MAE (Mean Absolute Error), R² Score

---

## 技術棧
- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, seaborn
- json (處理 JSON 欄位)

---

## 時間規劃
- Phase 1-2: 資料理解與 EDA (預計 2-3 小時)
- Phase 3: 資料預處理 (預計 3-4 小時)
- Phase 4: 模型訓練 (預計 4-5 小時)
- Phase 5: 模型融合 (預計 2-3 小時)
- Phase 6-7: 評估與提交 (預計 1-2 小時)

---

## 專案完成狀態

### ✅ 所有 Phase 已完成

**總體進度**: 100% ✅

- Phase 1: 問題定義與資料理解 ✅
- Phase 2: 探索性資料分析 ✅
- Phase 3: 資料預處理 ✅
- Phase 4: 模型建立與訓練 ✅
- Phase 5: 模型融合 ✅
- Phase 6: 評估與優化 ✅
- Phase 7: 預測與提交 ✅

### 📊 產出成果

**程式檔案**:
- ✅ simple_main.py（簡化版）
- ✅ run_complete_pipeline.py（完整版）
- ✅ main-2.py（Kaggle 版本）⭐
- ✅ eda_visualization.py（EDA 視覺化）

**文件檔案**:
- ✅ task.md（本文件）
- ✅ report.md（完整報告）
- ✅ README.md（專案說明）
- ✅ KAGGLE_GUIDE.md（Kaggle 指南）
- ✅ PROJECT_SUMMARY.md（專案總結）

**輸出檔案**:
- ✅ submission.csv（預測結果）
- ✅ model_evaluation_report.png（評估報告）
- ✅ eda_*.png（6 張 EDA 圖表）

### 🎯 最終成果

- **模型性能**: R² = 0.6819, RMSE = $73.1M
- **特徵數量**: 32 個完整特徵
- **模型數量**: 4 個（RF, XGBoost, LightGBM, CatBoost）
- **融合策略**: Voting Ensemble
- **文件完整度**: 100%

## 備註
- ✅ 已處理 JSON 格式的文本欄位
- ✅ 已實作 log transformation 特徵
- ✅ 已處理 budget=0 的情況
- ✅ 已完成交叉驗證
- ✅ 已生成完整的視覺化報告
