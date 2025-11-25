# Homework-2: Ensemble Learning

這是一個關於 Ensemble Learning 的機器學習作業實作，使用 MNIST 和 Fashion MNIST 資料集進行多類別分類。

## 📋 作業要求

### Part 1: MNIST Dataset
- 載入並分割資料集（50,000 訓練 / 10,000 驗證 / 10,000 測試）
- 訓練多種分類器（Random Forest, Extra Trees, SVM）
- 使用 Soft Voting 建立 Ensemble
- 評估並比較效能

### Part 2: Fashion MNIST Dataset
- 使用相同方法在 Fashion MNIST 上實驗
- 討論效能表現

## 🚀 快速開始

### 安裝相依套件

```bash
pip install numpy scikit-learn
```

### 執行主程式

```bash
python homework2_ensemble.py

#Final version with bonus methods
python FIN_homework2_ensemble.py

```

這會執行完整的實驗，包括：
- MNIST 資料集的訓練和評估
- Fashion MNIST 資料集的訓練和評估
- 效能比較和分析

## 📁 檔案結構

```
.
├── homework2_ensemble.py          # 主程式（必做部分）
├── bonus_ensemble_methods.py      # Bonus 進階方法
├── FIN_homework2_ensemble.py      # 最終版本 with bonus methods & 混淆矩陣
├── task.md                        # 作業記錄文件
└── README.md                      # 本檔案
```

## 🎯 主要功能

### homework2_ensemble.py

主程式包含以下功能：

1. **資料載入與分割**
   ```python
   load_and_split_mnist()
   load_and_split_fashion_mnist()
   ```

2. **訓練個別分類器**
   ```python
   train_individual_classifiers_mnist()
   ```
   - Random Forest (100 estimators)
   - Extra Trees (100 estimators)
   - SVM (RBF kernel)

3. **建立 Soft Voting Ensemble**
   ```python
   create_soft_voting_ensemble()
   ```

4. **評估與比較**
   ```python
   evaluate_ensemble_mnist()
   ```

### bonus_ensemble_methods.py

進階 Ensemble 方法（Bonus 部分）：

1. **Stacking Ensemble**
   - 使用 Logistic Regression 作為 meta-learner
   - 5-fold cross-validation

2. **Weighted Voting**
   - 根據驗證集效能分配權重
   - 效能好的分類器獲得更高權重

3. **Gradient Boosting**
   - 使用 Gradient Boosting Classifier
   - 100 estimators, learning rate 0.1

## 📊 預期輸出

程式執行後會顯示：

```
==================================================================
HOMEWORK 2: ENSEMBLE LEARNING
==================================================================

### PART 1: MNIST DATASET ###

Loading MNIST dataset...
Training set: (50000, 784)
Validation set: (10000, 784)
Test set: (10000, 784)

==================================================================
Training Individual Classifiers on MNIST
==================================================================

[1/3] Training Random Forest...
Random Forest - Validation Accuracy: 0.9XXX (Time: XX.XXs)

[2/3] Training Extra Trees...
Extra Trees - Validation Accuracy: 0.9XXX (Time: XX.XXs)

[3/3] Training SVM...
SVM - Validation Accuracy: 0.9XXX (Time: XX.XXs)

==================================================================
Creating Soft Voting Ensemble
==================================================================

Training Soft Voting Ensemble...
Ensemble - Validation Accuracy: 0.9XXX
Ensemble - Test Accuracy: 0.9XXX

==================================================================
Performance Comparison on Test Set
==================================================================
Random Forest        - Val: 0.9XXX, Test: 0.9XXX
Extra Trees          - Val: 0.9XXX, Test: 0.9XXX
SVM                  - Val: 0.9XXX, Test: 0.9XXX
Soft Voting Ensemble - Val: 0.9XXX, Test: 0.9XXX

Improvement over best individual: 0.0XXX (X.XX%)

### PART 2: FASHION MNIST DATASET ###
...
```

## 🔬 技術細節

### Soft Voting 原理

Soft Voting 使用每個分類器的預測機率：

1. 每個分類器輸出類別機率分布
2. 計算所有分類器的平均機率
3. 選擇平均機率最高的類別

數學表示：
```
P(class=c) = (1/N) * Σ P_i(class=c)
```

其中 N 是分類器數量，P_i 是第 i 個分類器的預測機率。

### 為什麼 Ensemble 效果更好？

1. **降低 Variance**: 多個模型的平均減少過擬合
2. **互補性**: 不同演算法捕捉不同的模式
3. **錯誤糾正**: 單一模型的錯誤可能被其他模型糾正
4. **穩定性**: 對資料變化更穩健

## 🎁 Bonus 實作

要使用 Bonus 方法，可以在主程式中加入：

```python
from bonus_ensemble_methods import run_bonus_experiments

# 在訓練完個別分類器後
bonus_results = run_bonus_experiments(
    X_train, y_train, X_val, y_val, X_test, y_test, classifiers
)
```

## ⚙️ 參數調整

可以調整的參數：

- **n_estimators**: Random Forest 和 Extra Trees 的樹數量
- **kernel**: SVM 的核函數（'rbf', 'linear', 'poly'）
- **voting**: 投票方式（'soft' 或 'hard'）
- **random_state**: 隨機種子（確保可重現性）

## 📝 注意事項

1. **執行時間**: 完整執行可能需要 10-30 分鐘，取決於硬體
2. **記憶體**: 需要至少 4GB RAM
3. **SVM 訓練**: 由於計算成本，使用部分資料訓練
4. **Bonus 方法**: 為了加速，使用較少的訓練資料

## 🔍 效能優化建議

如果執行太慢，可以：

1. 減少 estimators 數量（例如從 100 降到 50）
2. 使用更少的訓練資料
3. 移除 SVM（最慢的分類器）
4. 使用 `n_jobs=-1` 啟用平行處理

## 📚 參考資料

- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Voting Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)


