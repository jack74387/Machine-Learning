# Kaggle 快速開始指南

## 🚀 3 步驟提交到 Kaggle

### 步驟 1: 創建 Notebook
1. 前往 https://www.kaggle.com/competitions/tmdb-box-office-prediction
2. 點擊 "Code" → "New Notebook"
3. 選擇 Python 環境

### 步驟 2: 複製程式碼
1. 打開本地的 `kaggle_notebook.py` 檔案
2. **全選並複製**所有內容 (Ctrl+A, Ctrl+C)
3. 在 Kaggle Notebook 中**貼上** (Ctrl+V)

### 步驟 3: 執行並提交
1. 點擊 "Run All" 或按 Shift+Enter 執行
2. 等待執行完成（約 2-3 分鐘）
3. 在 Output 中找到 `submission.csv`
4. 點擊 "Submit to Competition"

---

## ✅ 檔案說明

### 推薦使用（按優先順序）

1. **`kaggle_notebook.py`** ⭐⭐⭐
   - ✅ 專為 Kaggle 設計
   - ✅ 已測試無語法錯誤
   - ✅ 支援多模型融合
   - ✅ 32 個完整特徵
   - **推薦指數**: ⭐⭐⭐⭐⭐

2. **`run_complete_pipeline.py`** ⭐⭐
   - ✅ 本地完整流程
   - ✅ 包含視覺化
   - ⚠️ 需要本地資料路徑
   - **推薦指數**: ⭐⭐⭐⭐ (本地使用)

3. **`simple_main.py`** ⭐
   - ✅ 快速測試
   - ⚠️ 只有基礎特徵
   - ⚠️ 單一模型
   - **推薦指數**: ⭐⭐⭐ (學習用)

### ⚠️ 不推薦使用

- ~~`main-2.py`~~ - 可能有複製貼上問題
- ~~`main.py`~~ - 需要手動修改路徑

---

## 📋 執行檢查清單

在 Kaggle 執行前，確認：

- [ ] 使用 `kaggle_notebook.py`
- [ ] 完整複製所有程式碼
- [ ] 沒有修改路徑（已設定為 Kaggle 路徑）
- [ ] 選擇 Python 環境
- [ ] 有足夠的執行時間（建議 GPU 環境）

---

## 🔧 常見問題排除

### Q: 出現 "SyntaxError: unterminated string literal"

**A**: 複製貼上時出現問題，請：
1. 重新打開 `kaggle_notebook.py`
2. 確保完整複製（從第一行到最後一行）
3. 在 Kaggle 中清空所有內容後再貼上

### Q: 出現 "FileNotFoundError"

**A**: 路徑設定錯誤，確認：
- 使用 `kaggle_notebook.py`（已設定正確路徑）
- 不要修改 TRAIN_PATH 和 TEST_PATH

### Q: 執行時間過長

**A**: 
- 切換到 GPU 環境（Settings → Accelerator → GPU）
- 或減少模型數量（註解掉部分模型）

### Q: 記憶體不足

**A**:
- 減少 n_estimators（從 200 改為 100）
- 減少特徵數量
- 使用較小的驗證集（test_size=0.15）

---

## 📊 預期結果

### 執行時間
- **CPU**: 約 3-5 分鐘
- **GPU**: 約 2-3 分鐘

### 模型性能
- **Random Forest**: R² ≈ 0.68-0.70
- **XGBoost**: R² ≈ 0.72-0.75
- **LightGBM**: R² ≈ 0.72-0.75
- **CatBoost**: R² ≈ 0.72-0.75
- **Voting Ensemble**: R² ≈ 0.75-0.78

### Kaggle 分數
- **Public Leaderboard**: RMSE ≈ $2.5-3.0M (log scale)
- **排名**: 預期 Top 30-50%

---

## 🎯 優化建議

### 快速優化（5-10 分鐘）
1. 調整 `n_estimators` (100 → 300)
2. 調整 `max_depth` (20 → 25)
3. 調整 `learning_rate` (0.05 → 0.03)

### 進階優化（30-60 分鐘）
1. 添加更多特徵工程
2. 使用 GridSearchCV 調參
3. 實作 Stacking Ensemble
4. 對目標變數進行 log 轉換

---

## 📝 提交後

### 檢查結果
1. 查看 Public Leaderboard 分數
2. 比較與 Baseline 的差異
3. 分析錯誤案例

### 迭代改進
1. 根據排行榜反饋調整
2. 嘗試不同的特徵組合
3. 調整模型參數
4. 嘗試不同的融合策略

---

## 🎉 成功標準

- ✅ 程式成功執行
- ✅ 生成 submission.csv
- ✅ 成功提交到 Kaggle
- ✅ Public Leaderboard 有分數
- ✅ 分數優於 Baseline

---

## 📞 需要幫助？

如果遇到問題：
1. 檢查 [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) 詳細說明
2. 查看 [FINAL_CHECKLIST.md](FINAL_CHECKLIST.md) 完整檢查清單
3. 參考 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) 專案總結

---

**祝你在 Kaggle 競賽中取得好成績！** 🏆
