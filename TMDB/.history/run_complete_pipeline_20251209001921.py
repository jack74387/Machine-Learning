"""
TMDB 電影票房預測 - 完整流程
整合 EDA、特徵工程、模型訓練和預測
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print(" " * 15 + "TMDB 電影票房預測 - 完整流程")
print("=" * 70)

# ============================================================================
# Phase 1 & 2: 資料載入與探索
# ============================================================================
print("\n【Phase 1-2】資料載入與探索性分析")
print("-" * 70)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"✓ 訓練集: {train_df.shape}")
print(f"✓ 測試集: {test_df.shape}")

print(f"\n目標變數統計:")
print(f"  平均值: ${train_df['revenue'].mean():,.0f}")
print(f"  中位數: ${train_df['revenue'].median():,.0f}")
print(f"  最大值: ${train_df['revenue'].max():,.0f}")

# 關鍵相關性
corr_budget = train_df[['budget', 'revenue']].corr().iloc[0, 1]
corr_popularity = train_df[['popularity', 'revenue']].corr().iloc[0, 1]
print(f"\n關鍵相關性:")
print(f"  Budget vs Revenue: {corr_budget:.3f}")
print(f"  Popularity vs Revenue: {corr_popularity:.3f}")

# ============================================================================
# Phase 3: 特徵工程
# ============================================================================
print("\n【Phase 3】特徵工程")
print("-" * 70)

def parse_json_count(x):
    """解析 JSON 並返回數量"""
    try:
        if pd.isna(x) or x == '':
            return 0
        data = json.loads(x.replace("'", '"'))
        return len(data)
    except:
        return 0

def parse_json_first_name(x):
    """解析 JSON 並返回第一個 name"""
    try:
        if pd.isna(x) or x == '':
            return 'Unknown'
        data = json.loads(x.replace("'", '"'))
        if isinstance(data, list) and len(data) > 0:
            return data[0].get('name', 'Unknown')
        return 'Unknown'
    except:
        return 'Unknown'

def feature_engineering(df, label_encoders=None, is_train=True):
    """特徵工程"""
    df = df.copy()
    
    print(f"  處理 {'訓練' if is_train else '測試'}集...")
    
    # 1. 數值特徵
    df['budget'] = df['budget'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # 2. 時間特徵
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
    
    # 3. JSON 欄位
    df['genres_count'] = df['genres'].apply(parse_json_count)
    df['top_genre'] = df['genres'].apply(parse_json_first_name)
    df['cast_count'] = df['cast'].apply(parse_json_count)
    df['crew_count'] = df['crew'].apply(parse_json_count)
    df['keywords_count'] = df['Keywords'].apply(parse_json_count)
    df['production_companies_count'] = df['production_companies'].apply(parse_json_count)
    
    # 4. 系列電影
    df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
    
    # 5. 文本長度
    df['title_length'] = df['title'].fillna('').apply(len)
    df['overview_length'] = df['overview'].fillna('').apply(len)
    df['tagline_length'] = df['tagline'].fillna('').apply(len)
    
    # 6. 衍生特徵
    df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)
    df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
    
    # 7. 類別特徵編碼
    df['original_language'] = df['original_language'].fillna('en')
    df['status'] = df['status'].fillna('Released')
    df['top_genre'] = df['top_genre'].fillna('Unknown')
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in ['original_language', 'status', 'top_genre']:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            def safe_transform(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                else:
                    return le.transform([le.classes_[0]])[0]
            df[col] = df[col].astype(str).apply(safe_transform)
    
    return df, label_encoders

# 執行特徵工程
train_processed, label_encoders = feature_engineering(train_df, is_train=True)
test_processed, _ = feature_engineering(test_df, label_encoders=label_encoders, is_train=False)

print(f"✓ 特徵工程完成")

# ============================================================================
# Phase 4: 準備訓練資料
# ============================================================================
print("\n【Phase 4】準備訓練資料")
print("-" * 70)

feature_cols = [
    'budget', 'popularity', 'runtime',
    'release_year', 'release_month', 'release_quarter',
    'genres_count', 'cast_count', 'crew_count', 'keywords_count',
    'production_companies_count', 'has_collection',
    'title_length', 'overview_length', 'tagline_length',
    'budget_popularity_ratio', 'budget_per_minute',
    'original_language', 'status', 'top_genre'
]

X = train_processed[feature_cols]
y = train_processed['revenue']
X_test = test_processed[feature_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"✓ 訓練集: {X_train.shape}")
print(f"✓ 驗證集: {X_val.shape}")
print(f"✓ 測試集: {X_test.shape}")
print(f"✓ 特徵數量: {len(feature_cols)}")

# ============================================================================
# Phase 5: 模型訓練
# ============================================================================
print("\n【Phase 5】模型訓練")
print("-" * 70)

print("  訓練 Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("✓ 模型訓練完成")

# ============================================================================
# Phase 6: 模型評估
# ============================================================================
print("\n【Phase 6】模型評估")
print("-" * 70)

# 訓練集評估
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"訓練集性能:")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE:  ${train_mae:,.2f}")
print(f"  R²:   {train_r2:.4f}")

# 驗證集評估
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n驗證集性能:")
print(f"  RMSE: ${val_rmse:,.2f}")
print(f"  MAE:  ${val_mae:,.2f}")
print(f"  R²:   {val_r2:.4f}")

# 交叉驗證
print(f"\n執行 5-Fold 交叉驗證...")
cv_scores = cross_val_score(model, X, y, cv=5, 
                            scoring='neg_root_mean_squared_error',
                            n_jobs=-1)
cv_rmse = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"  交叉驗證 RMSE: ${cv_rmse:,.2f} (±${cv_std:,.2f})")

# 特徵重要性
print(f"\n【特徵重要性 Top 10】")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# Phase 7: 預測與提交
# ============================================================================
print("\n【Phase 7】生成預測結果")
print("-" * 70)

predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)  # 確保預測值為正數

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': predictions
})

submission.to_csv('submission.csv', index=False)
print("✓ 提交檔案已生成: submission.csv")

print(f"\n預測統計:")
print(f"  最小值: ${predictions.min():,.2f}")
print(f"  最大值: ${predictions.max():,.2f}")
print(f"  平均值: ${predictions.mean():,.2f}")
print(f"  中位數: ${np.median(predictions):,.2f}")

# ============================================================================
# 生成視覺化報告
# ============================================================================
print("\n【生成視覺化報告】")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 實際 vs 預測 (驗證集)
axes[0, 0].scatter(y_val, y_val_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Revenue ($)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Revenue ($)', fontsize=11)
axes[0, 0].set_title('Actual vs Predicted (Validation Set)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].text(0.05, 0.95, f'R² = {val_r2:.4f}\nRMSE = ${val_rmse:,.0f}', 
                transform=axes[0, 0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top', fontsize=10)

# 2. 殘差分布
residuals = y_val - y_val_pred
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals ($)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

# 3. 特徵重要性
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'], fontsize=9)
axes[1, 0].set_xlabel('Importance', fontsize=11)
axes[1, 0].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. 預測分布
axes[1, 1].hist(predictions, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 1].axvline(predictions.mean(), color='red', linestyle='--', 
                   label=f'Mean: ${predictions.mean():,.0f}')
axes[1, 1].axvline(np.median(predictions), color='blue', linestyle='--', 
                   label=f'Median: ${np.median(predictions):,.0f}')
axes[1, 1].set_xlabel('Predicted Revenue ($)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Prediction Distribution (Test Set)', fontsize=12, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_evaluation_report.png', dpi=300, bbox_inches='tight')
print("✓ 視覺化報告已儲存: model_evaluation_report.png")
plt.close()

# ============================================================================
# 最終總結
# ============================================================================
print("\n" + "=" * 70)
print(" " * 25 + "專案完成總結")
print("=" * 70)

print("\n【模型性能】")
print(f"  驗證集 RMSE: ${val_rmse:,.2f}")
print(f"  驗證集 R²:   {val_r2:.4f}")
print(f"  交叉驗證 RMSE: ${cv_rmse:,.2f}")

print("\n【關鍵特徵】")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']:25s} ({row['importance']:.2%})")

print("\n【生成檔案】")
print("  ✓ submission.csv - 預測結果")
print("  ✓ model_evaluation_report.png - 評估報告")
print("  ✓ eda_*.png - EDA 圖表 (6 張)")

print("\n【下一步建議】")
print("  1. 安裝完整套件 (pip install -r requirements.txt)")
print("  2. 執行 main.py 使用多模型融合")
print("  3. 調整超參數以提升性能")
print("  4. 嘗試更多特徵工程")

print("\n" + "=" * 70)
print(" " * 20 + "感謝使用 Taskmaster 方法！")
print("=" * 70)
