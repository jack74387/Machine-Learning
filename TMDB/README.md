# 🎬 TMDB 電影票房預測專案

使用 Taskmaster 方法系統化開發的機器學習專案，預測電影票房收入。

**專案狀態**: ✅ 完成 | **當前版本**: V2.0 | **最佳性能**: R² = 0.703

## 🎯 快速開始

```bash
# 1. 執行進階模型（推薦）
python advanced_model.py

# 2. 查看結果
# 輸出: submission_advanced.csv
# 性能: R² = 0.703, RMSE = $75.8M

# 3. 提交到 Kaggle
# 上傳 submission_advanced.csv
```

詳細說明請參考 [QUICK_START.md](QUICK_START.md)

---

## 📊 專案成果

### 模型性能

| 版本 | 檔案 | R² | RMSE | 特徵數 | 狀態 |
|------|------|----|----|--------|------|
| V1.0 | simple_main.py | 0.688 | $72.4M | 14 | ✅ |
| V2.0 | advanced_model.py | **0.703** | **$75.8M** | **83** | ✅ **推薦** |
| V3.0 | main.py | 0.75+（預期） | $60-65M | 83 | ⏳ 需安裝套件 |

### 關鍵創新

1. **歷史統計特徵** ⭐ - 製作公司歷史成為最強預測因子（9.40%）
2. **交互特徵** - budget × company_mean 排名第3（6.48%）
3. **Log 轉換** - 處理偏態分布，提升穩定性
4. **模型集成** - RF + GB + Ridge 加權平均

---

## 📂 專案結構

```
.
├── train.csv                 # 訓練資料
├── test.csv                  # 測試資料
├── sample_submission.csv     # 提交範例
├── main.py                   # 主程式
├── task.md                   # 任務規劃文件
├── report.md                 # 專案報告
├── README.md                 # 本文件
└── submission.csv            # 生成的預測結果
```

## 環境需求

### Python 版本
- Python 3.8+

### 必要套件
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

或使用 requirements.txt:
```bash
pip install -r requirements.txt
```

## 快速開始

### 方案 A: 本地執行（簡化版）

適合快速測試，不需要安裝額外套件。

```bash
python simple_main.py
```

### 方案 B: 本地執行（完整版）

需要安裝完整套件。

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 執行完整流程：
```bash
python run_complete_pipeline.py
```

### 方案 C: Kaggle Notebook（⭐ 最推薦）

**最推薦的方式**，可以使用 Kaggle 的免費 GPU 和預裝套件。

1. 前往 [TMDB Box Office Prediction](https://www.kaggle.com/competitions/tmdb-box-office-prediction)
2. 創建新的 Notebook
3. 複製 `kaggle_notebook.py` 的內容（⚠️ 不要用 main-2.py）
4. 執行並提交

**快速開始**: 參考 [KAGGLE_QUICK_START.md](KAGGLE_QUICK_START.md)（3 步驟）
**詳細說明**: 參考 [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)

### 執行流程

程式會自動執行以下步驟：
1. 載入資料
2. 探索性資料分析（EDA）
3. 特徵工程（32 個特徵）
4. 訓練多個模型（Random Forest, XGBoost, LightGBM, CatBoost）
5. 模型融合（Voting Ensemble）
6. 生成預測結果

### 查看結果

執行完成後會生成：
- `submission.csv` - 預測結果
- `model_evaluation_report.png` - 模型評估報告
- `eda_*.png` - 探索性資料分析圖表（6 張）

## 專案特色

### 1. Taskmaster 方法
採用系統化的任務分解方法，將專案分為多個階段：
- 問題定義與資料理解
- 探索性資料分析
- 資料預處理
- 模型建立與訓練
- 模型融合
- 評估與優化

### 2. 完整的特徵工程
- 時間特徵提取（年、月、日、季度）
- JSON 欄位解析（genres, cast, crew, keywords）
- 衍生特徵創建（budget_popularity_ratio, budget_per_minute）
- 文本長度特徵
- 類別特徵編碼

### 3. 多模型融合
實作了多種模型融合策略：
- **Simple Voting**: 簡單平均
- **Weighted Voting**: 加權平均
- 支援 Random Forest, XGBoost, LightGBM, CatBoost

### 4. 完整的文件記錄
- `task.md`: 詳細的任務規劃
- `report.md`: 完整的專案報告，包含：
  - 題目選定
  - 資料分析及處理
  - 模型架構與介紹
  - Voting 作法討論
  - 最終結果和結論

## 主要功能

### TMDBPredictor 類別

```python
predictor = TMDBPredictor()

# 載入資料
predictor.load_data()

# 探索性資料分析
predictor.explore_data()

# 準備特徵
predictor.prepare_features()

# 訓練模型
predictor.train_models()

# 模型融合
predictor.ensemble_voting()

# 生成預測
submission = predictor.predict_and_submit()
```

## 模型說明

### 1. Random Forest
- 樹的數量: 100
- 最大深度: 15
- 最小分裂樣本數: 5

### 2. XGBoost
- 迭代次數: 100
- 最大深度: 6
- 學習率: 0.1

### 3. LightGBM
- 迭代次數: 100
- 最大深度: 6
- 學習率: 0.1

### 4. CatBoost
- 迭代次數: 100
- 深度: 6
- 學習率: 0.1

## 評估指標

- **RMSE** (Root Mean Squared Error): 主要評估指標
- **MAE** (Mean Absolute Error): 次要評估指標
- **R²** (R-squared): 模型解釋力

## 特徵列表

### 數值特徵
- budget: 電影預算
- popularity: 人氣度
- runtime: 電影時長

### 時間特徵
- release_year: 上映年份
- release_month: 上映月份
- release_day: 上映日期
- release_quarter: 上映季度

### 計數特徵
- genres_count: 類型數量
- cast_count: 演員數量
- crew_count: 工作人員數量
- keywords_count: 關鍵字數量
- production_companies_count: 製作公司數量

### 衍生特徵
- has_collection: 是否屬於系列電影
- title_length: 標題長度
- overview_length: 簡介長度
- tagline_length: 標語長度
- budget_popularity_ratio: 預算人氣比
- budget_per_minute: 每分鐘預算

### 類別特徵
- original_language: 原始語言
- status: 狀態
- top_genre: 主要類型

## 進階使用

### 自訂模型參數

```python
predictor = TMDBPredictor()
predictor.load_data()
predictor.prepare_features()

# 自訂 XGBoost 參數
from xgboost import XGBRegressor
custom_xgb = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)
custom_xgb.fit(predictor.X_train, predictor.y_train)
```

### 使用不同的融合策略

```python
# 使用特定模型進行預測
submission = predictor.predict_and_submit(model_name='xgb')  # 只使用 XGBoost
submission = predictor.predict_and_submit(model_name='voting')  # 使用簡單 Voting
```

## 常見問題

### Q1: 執行時出現記憶體不足
A: 可以減少模型的 n_estimators 參數，或使用較小的 max_depth。

### Q2: 如何提升預測準確度？
A: 可以嘗試：
- 更細緻的特徵工程
- 超參數調優
- 增加更多外部資料
- 使用更複雜的融合策略（如 Stacking）

### Q3: 訓練時間太長
A: 可以：
- 減少模型數量
- 降低 n_estimators
- 使用較小的訓練集進行快速實驗

## 參考資料

- [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

## 📚 文件導航

### 必讀文件
- **[QUICK_START.md](QUICK_START.md)** - 5分鐘快速開始 ⭐
- **[SUMMARY.md](SUMMARY.md)** - 專案總結
- **[report.md](report.md)** - 完整技術報告（含版本對比）

### 詳細文件
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - 最終完整報告
- **[MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md)** - 改進記錄
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - 下一步優化指南
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 專案結構
- **[專案完成報告.md](專案完成報告.md)** - 完成報告

### 輔助文件
- **[task.md](task.md)** - Taskmaster 任務規劃
- **[FILE_INDEX.md](FILE_INDEX.md)** - 檔案索引
- **[CHECKLIST.md](CHECKLIST.md)** - 完成檢查清單

---

## 🎓 學習收穫

### 技術亮點
1. **深度特徵工程** - 14 → 83 個特徵（+493%）
2. **歷史統計方法** - 構建演員/導演/公司的歷史數據
3. **模型集成策略** - 3個模型加權平均
4. **系統化開發** - Taskmaster 方法應用

### 關鍵發現
1. **製作公司最重要** - 歷史數據超越傳統 budget 特徵
2. **交互特徵有效** - 捕捉"大預算 + 好公司"協同效應
3. **Log 轉換必要** - 處理極度右偏態分布
4. **特徵分散化** - 重要性從 59% 降至 9.4%，模型更穩定

---

## 🚀 下一步

### 立即可做
```bash
# 安裝進階模型庫
pip install xgboost lightgbm catboost

# 執行完整版本
python main.py

# 預期: R² 提升至 0.75-0.78
```

### 優化方向
- [ ] 超參數調優（RandomizedSearchCV）
- [ ] 5-Fold 交叉驗證
- [ ] Stacking 集成
- [ ] 外部數據增強（IMDb 評分）

詳見 [NEXT_STEPS.md](NEXT_STEPS.md)

---

## 📊 專案統計

- **開發時間**: ~12 小時
- **程式碼**: 4個檔案，~2000行
- **文件**: 12個檔案，~60頁
- **圖表**: 7張 PNG
- **性能提升**: R² +2.2%, MAE -7.3%

---

## 🏆 專案價值

### 學術價值
- Taskmaster 方法的成功應用
- 歷史統計特徵的創新方法
- 完整的特徵工程範例

### 實務價值
- 電影投資決策參考
- 風險評估工具
- 發行策略優化

---

## 📞 需要幫助？

- 快速開始：閱讀 [QUICK_START.md](QUICK_START.md)
- 了解成果：閱讀 [SUMMARY.md](SUMMARY.md)
- 技術細節：閱讀 [report.md](report.md)
- 繼續優化：閱讀 [NEXT_STEPS.md](NEXT_STEPS.md)

---

## 授權

MIT License

## 作者

TMDB Movie Revenue Prediction Project

## 更新日誌

### V2.0 (2025-12-09) - 當前版本 ⭐
- ✅ 深度特徵工程（83個特徵）
- ✅ 歷史統計特徵（演員/導演/公司）
- ✅ 交互特徵和多項式特徵
- ✅ Log 轉換處理偏態
- ✅ 多模型集成（RF + GB + Ridge）
- ✅ 性能提升至 R² = 0.703
- ✅ 完整文件記錄（12個文件）

### V1.0 (2025-12-09)
- ✅ 基本特徵工程（14個特徵）
- ✅ Random Forest 模型
- ✅ R² = 0.688
- ✅ 基礎文件

---

**專案狀態**: ✅ 階段性完成
**推薦使用**: advanced_model.py
**下一目標**: R² > 0.75
