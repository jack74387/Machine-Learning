# 下一步優化指南

## 🎯 當前狀態
- **最佳模型**: Gradient Boosting
- **R² 分數**: 0.7034
- **RMSE**: $75,770,328
- **特徵數量**: 83個

## 🚀 立即可做（5分鐘內）

### 1. 提交到 Kaggle
```bash
# 使用進階模型的預測結果
# 檔案: submission_advanced.csv
```

### 2. 安裝進階模型庫
```bash
pip install xgboost lightgbm catboost
```

然後重新執行：
```bash
python advanced_model.py
```

**預期改進**:
- R² 提升至 0.75-0.78
- RMSE 降低至 $60-65M
- Kaggle 排名提升 20-30%

---

## 📈 短期優化（1-2小時）

### 1. 超參數調優

創建 `hyperparameter_tuning.py`:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# 定義參數空間
param_dist = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [5, 6, 7, 8, 9],
    'learning_rate': [0.01, 0.02, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# 隨機搜索
random_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)
print(f"最佳參數: {random_search.best_params_}")
print(f"最佳分數: {random_search.best_score_}")
```

**預期改進**: R² +0.01-0.02

### 2. 5-Fold 交叉驗證

修改 `advanced_model.py` 添加：

```python
from sklearn.model_selection import cross_val_score

# 5-Fold CV
cv_scores = cross_val_score(
    model, X, y_log, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)

print(f"CV R² 分數: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**預期改進**: 更穩定的評估，減少過擬合

### 3. 特徵選擇

```python
from sklearn.feature_selection import SelectFromModel

# 使用特徵重要性選擇
selector = SelectFromModel(
    GradientBoostingRegressor(n_estimators=100),
    threshold='median'
)
selector.fit(X_train, y_train)

# 獲取選中的特徵
selected_features = X.columns[selector.get_support()]
print(f"選中 {len(selected_features)} 個特徵")
```

**預期改進**: 減少過擬合，提升泛化能力

---

## 🎨 中期優化（1天）

### 1. Stacking 集成

```python
from sklearn.ensemble import StackingRegressor

# 基礎模型
estimators = [
    ('rf', RandomForestRegressor(...)),
    ('gb', GradientBoostingRegressor(...)),
    ('xgb', xgb.XGBRegressor(...)),
    ('lgb', lgb.LGBMRegressor(...))
]

# 元模型
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=10),
    cv=5
)

stacking.fit(X_train, y_train)
```

**預期改進**: R² +0.02-0.03

### 2. 更多特徵工程

#### A. 文本特徵（TF-IDF）
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Overview 的 TF-IDF
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
overview_tfidf = tfidf.fit_transform(df['overview'].fillna(''))
```

#### B. 時間序列特徵
```python
# 同年上映電影數量
df['movies_same_year'] = df.groupby('release_year')['id'].transform('count')

# 同月上映電影平均票房
df['avg_revenue_same_month'] = df.groupby('release_month')['revenue'].transform('mean')
```

#### C. 網絡特徵
```python
# 演員合作次數
actor_pairs = {}
for cast_list in df['cast_list']:
    for i in range(len(cast_list)):
        for j in range(i+1, len(cast_list)):
            pair = tuple(sorted([cast_list[i]['name'], cast_list[j]['name']]))
            actor_pairs[pair] = actor_pairs.get(pair, 0) + 1
```

**預期改進**: R² +0.01-0.02

### 3. 異常值處理

```python
# 識別異常值
Q1 = df['revenue'].quantile(0.25)
Q3 = df['revenue'].quantile(0.75)
IQR = Q3 - Q1

# 移除極端異常值（可選）
df_clean = df[
    (df['revenue'] >= Q1 - 3*IQR) & 
    (df['revenue'] <= Q3 + 3*IQR)
]

# 或使用 Robust Scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
```

**預期改進**: 提升穩定性

---

## 🔬 長期優化（1週+）

### 1. 外部數據增強

#### IMDb 評分
```python
# 需要爬蟲或 API
df['imdb_rating'] = df['imdb_id'].apply(get_imdb_rating)
df['imdb_votes'] = df['imdb_id'].apply(get_imdb_votes)
```

#### Rotten Tomatoes
```python
df['rt_score'] = df['title'].apply(get_rt_score)
df['rt_audience_score'] = df['title'].apply(get_rt_audience)
```

#### 社交媒體
```python
# Twitter/Facebook 提及次數
df['social_mentions'] = df['title'].apply(get_social_mentions)
```

**預期改進**: R² +0.03-0.05

### 2. 深度學習模型

```python
import tensorflow as tf
from tensorflow import keras

# 構建神經網絡
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
```

**預期改進**: 可能 +0.02-0.04（需要大量調優）

### 3. AutoML

```python
# 使用 H2O AutoML
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train_h2o = h2o.H2OFrame(train_df)

aml = H2OAutoML(max_runtime_secs=3600, seed=42)
aml.train(x=feature_cols, y='revenue', training_frame=train_h2o)

# 獲取最佳模型
best_model = aml.leader
```

**預期改進**: 自動找到最佳模型組合

---

## 📊 評估改進效果

### 創建評估腳本

```python
# evaluate_improvements.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_submission(submission_file, validation_file):
    """評估提交檔案"""
    sub = pd.read_csv(submission_file)
    val = pd.read_csv(validation_file)
    
    merged = pd.merge(sub, val, on='id')
    
    rmse = np.sqrt(mean_squared_error(merged['revenue_true'], merged['revenue_pred']))
    r2 = r2_score(merged['revenue_true'], merged['revenue_pred'])
    
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²: {r2:.4f}")
    
    return rmse, r2

# 比較不同版本
versions = {
    'V1.0 基礎': 'submission.csv',
    'V2.0 進階': 'submission_advanced.csv',
    'V3.0 優化': 'submission_optimized.csv'
}

for name, file in versions.items():
    print(f"\n{name}:")
    evaluate_submission(file, 'validation.csv')
```

---

## 🎯 目標設定

### 短期目標（本週）
- [ ] R² > 0.75
- [ ] RMSE < $70M
- [ ] Kaggle 排名進入前 50%

### 中期目標（本月）
- [ ] R² > 0.78
- [ ] RMSE < $65M
- [ ] Kaggle 排名進入前 30%

### 長期目標（本季）
- [ ] R² > 0.80
- [ ] RMSE < $60M
- [ ] Kaggle 排名進入前 10%

---

## 📝 實驗追蹤

創建 `experiments.csv` 追蹤所有實驗：

| 日期 | 版本 | 模型 | 特徵數 | R² | RMSE | 備註 |
|------|------|------|--------|----|----|------|
| 2025-12-09 | V1.0 | RF | 14 | 0.688 | $72.4M | 基礎版本 |
| 2025-12-09 | V2.0 | GB | 83 | 0.703 | $75.8M | 歷史特徵 |
| 2025-12-09 | V2.0 | Ensemble | 83 | 0.698 | $76.5M | 3模型集成 |
| ... | ... | ... | ... | ... | ... | ... |

---

## 🔧 調試技巧

### 1. 檢查預測分布
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(y_true, bins=50, alpha=0.7, label='True')
plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted')
plt.legend()
plt.title('Distribution Comparison')

plt.subplot(1, 3, 2)
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('True Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Prediction vs True')

plt.subplot(1, 3, 3)
residuals = y_true - y_pred
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Revenue')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.savefig('prediction_analysis.png')
```

### 2. 特徵相關性分析
```python
import seaborn as sns

# 計算相關性
corr_matrix = df[top_features + ['revenue']].corr()

# 視覺化
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('feature_correlation.png')
```

### 3. 學習曲線
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, 
    scoring='r2'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('R² Score')
plt.legend()
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
```

---

## 💡 專家建議

1. **不要過度擬合訓練集**
   - 始終關注驗證集性能
   - 使用交叉驗證
   - 早停（Early Stopping）

2. **特徵工程 > 模型選擇**
   - 好的特徵比複雜的模型更重要
   - 專注於領域知識
   - 創造有意義的交互特徵

3. **集成學習是王道**
   - 多樣化的模型組合
   - 不同類型的模型（樹模型 + 線性模型）
   - 適當的權重分配

4. **持續迭代**
   - 小步快跑
   - 記錄每次實驗
   - 從失敗中學習

---

## 📚 學習資源

- [Kaggle Learn - Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Kaggle Learn - Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)

---

**記住**: 機器學習是一個迭代過程。每次改進都是進步！🚀
