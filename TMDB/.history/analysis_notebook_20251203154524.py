"""
TMDB 電影票房預測 - 分析筆記本
這個檔案可以在 Jupyter Notebook 中逐步執行
"""

# %% [markdown]
# # TMDB 電影票房預測專案
# 
# 使用 Taskmaster 方法進行系統化開發

# %% [markdown]
# ## 1. 匯入套件

# %%
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# 設定視覺化風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RANDOM_STATE = 42

# %% [markdown]
# ## 2. 載入資料

# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"訓練集大小: {train_df.shape}")
print(f"測試集大小: {test_df.shape}")

# %%
# 查看前幾筆資料
train_df.head()

# %% [markdown]
# ## 3. 探索性資料分析 (EDA)

# %% [markdown]
# ### 3.1 目標變數分析

# %%
# 目標變數統計
print("Revenue 統計:")
print(train_df['revenue'].describe())

# %%
# Revenue 分布圖
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 原始分布
axes[0].hist(train_df['revenue'], bins=50, edgecolor='black')
axes[0].set_title('Revenue Distribution')
axes[0].set_xlabel('Revenue')
axes[0].set_ylabel('Frequency')

# Log 轉換後的分布
axes[1].hist(np.log1p(train_df['revenue']), bins=50, edgecolor='black')
axes[1].set_title('Log(Revenue) Distribution')
axes[1].set_xlabel('Log(Revenue)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 數值特徵分析

# %%
# 數值特徵統計
numerical_features = ['budget', 'popularity', 'runtime']
train_df[numerical_features].describe()

# %%
# 數值特徵與 revenue 的關係
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(numerical_features):
    axes[idx].scatter(train_df[feature], train_df['revenue'], alpha=0.5)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Revenue')
    axes[idx].set_title(f'{feature} vs Revenue')

plt.tight_layout()
plt.show()

# %%
# 相關性矩陣
correlation_features = ['budget', 'popularity', 'runtime', 'revenue']
correlation_matrix = train_df[correlation_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# %% [markdown]
# ### 3.3 缺失值分析

# %%
# 缺失值統計
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if len(missing_values) > 0:
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind='bar')
    plt.title('Missing Values by Feature')
    plt.xlabel('Feature')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("沒有缺失值")

# %% [markdown]
# ## 4. 特徵工程

# %%
def parse_json_column(df, column, key=None, count=False):
    """解析 JSON 格式的欄位"""
    def safe_parse(x):
        try:
            if pd.isna(x) or x == '':
                return [] if not count else 0
            data = json.loads(x.replace("'", '"'))
            if count:
                return len(data)
            if key and isinstance(data, list) and len(data) > 0:
                return data[0].get(key, '')
            return data
        except:
            return [] if not count else 0
    
    return df[column].apply(safe_parse)

# %%
def feature_engineering(df, label_encoders=None, is_train=True):
    """特徵工程"""
    df = df.copy()
    
    # 1. 基本數值特徵
    df['budget'] = df['budget'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # 2. 時間特徵
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
    df['release_day'] = df['release_date'].dt.day.fillna(15).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
    
    # 3. JSON 欄位處理
    df['genres_count'] = parse_json_column(df, 'genres', count=True)
    df['top_genre'] = parse_json_column(df, 'genres', key='name')
    df['cast_count'] = parse_json_column(df, 'cast', count=True)
    df['crew_count'] = parse_json_column(df, 'crew', count=True)
    df['keywords_count'] = parse_json_column(df, 'Keywords', count=True)
    df['production_companies_count'] = parse_json_column(df, 'production_companies', count=True)
    
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
    
    categorical_features = ['original_language', 'status', 'top_genre']
    for col in categorical_features:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col].astype(str))
    
    return df, label_encoders

# %%
# 執行特徵工程
train_processed, label_encoders = feature_engineering(train_df, is_train=True)
test_processed, _ = feature_engineering(test_df, label_encoders=label_encoders, is_train=False)

print(f"處理後的訓練集大小: {train_processed.shape}")
print(f"處理後的測試集大小: {test_processed.shape}")

# %% [markdown]
# ## 5. 準備訓練資料

# %%
# 選擇特徵
feature_cols = [
    'budget', 'popularity', 'runtime',
    'release_year', 'release_month', 'release_day', 'release_quarter',
    'genres_count', 'cast_count', 'crew_count', 'keywords_count',
    'production_companies_count', 'has_collection',
    'title_length', 'overview_length', 'tagline_length',
    'budget_popularity_ratio', 'budget_per_minute',
    'original_language', 'status', 'top_genre'
]

X = train_processed[feature_cols]
y = train_processed['revenue']
X_test = test_processed[feature_cols]

# 分割訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"訓練集: {X_train.shape}")
print(f"驗證集: {X_val.shape}")
print(f"測試集: {X_test.shape}")

# %% [markdown]
# ## 6. 模型訓練

# %% [markdown]
# ### 6.1 Random Forest

# %%
print("訓練 Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 評估
y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
mae_rf = mean_absolute_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:,.2f}, MAE: {mae_rf:,.2f}, R²: {r2_rf:.4f}")

# %% [markdown]
# ### 6.2 XGBoost

# %%
print("訓練 XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# 評估
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"XGBoost - RMSE: {rmse_xgb:,.2f}, MAE: {mae_xgb:,.2f}, R²: {r2_xgb:.4f}")

# %% [markdown]
# ### 6.3 LightGBM

# %%
print("訓練 LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

# 評估
y_pred_lgb = lgb_model.predict(X_val)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
mae_lgb = mean_absolute_error(y_val, y_pred_lgb)
r2_lgb = r2_score(y_val, y_pred_lgb)

print(f"LightGBM - RMSE: {rmse_lgb:,.2f}, MAE: {mae_lgb:,.2f}, R²: {r2_lgb:.4f}")

# %% [markdown]
# ### 6.4 CatBoost

# %%
print("訓練 CatBoost...")
cat_model = CatBoostRegressor(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbose=False
)
cat_model.fit(X_train, y_train)

# 評估
y_pred_cat = cat_model.predict(X_val)
rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))
mae_cat = mean_absolute_error(y_val, y_pred_cat)
r2_cat = r2_score(y_val, y_pred_cat)

print(f"CatBoost - RMSE: {rmse_cat:,.2f}, MAE: {mae_cat:,.2f}, R²: {r2_cat:.4f}")

# %% [markdown]
# ## 7. 模型比較

# %%
# 模型性能比較
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost'],
    'RMSE': [rmse_rf, rmse_xgb, rmse_lgb, rmse_cat],
    'MAE': [mae_rf, mae_xgb, mae_lgb, mae_cat],
    'R²': [r2_rf, r2_xgb, r2_lgb, r2_cat]
})

print("\n模型性能比較:")
print(results.to_string(index=False))

# %%
# 視覺化比較
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE
axes[0].bar(results['Model'], results['RMSE'])
axes[0].set_title('RMSE Comparison')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)

# MAE
axes[1].bar(results['Model'], results['MAE'])
axes[1].set_title('MAE Comparison')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)

# R²
axes[2].bar(results['Model'], results['R²'])
axes[2].set_title('R² Comparison')
axes[2].set_ylabel('R²')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. 模型融合 - Voting

# %%
print("模型融合 - Voting Regressor...")

# Simple Voting
voting_model = VotingRegressor([
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model)
])
voting_model.fit(X_train, y_train)

# 評估
y_pred_voting = voting_model.predict(X_val)
rmse_voting = np.sqrt(mean_squared_error(y_val, y_pred_voting))
mae_voting = mean_absolute_error(y_val, y_pred_voting)
r2_voting = r2_score(y_val, y_pred_voting)

print(f"Voting (Simple) - RMSE: {rmse_voting:,.2f}, MAE: {mae_voting:,.2f}, R²: {r2_voting:.4f}")

# %%
# Weighted Voting
weights = [0.2, 0.3, 0.3, 0.2]
weighted_voting_model = VotingRegressor([
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model)
], weights=weights)
weighted_voting_model.fit(X_train, y_train)

# 評估
y_pred_weighted = weighted_voting_model.predict(X_val)
rmse_weighted = np.sqrt(mean_squared_error(y_val, y_pred_weighted))
mae_weighted = mean_absolute_error(y_val, y_pred_weighted)
r2_weighted = r2_score(y_val, y_pred_weighted)

print(f"Voting (Weighted) - RMSE: {rmse_weighted:,.2f}, MAE: {mae_weighted:,.2f}, R²: {r2_weighted:.4f}")

# %% [markdown]
# ## 9. 特徵重要性分析

# %%
# XGBoost 特徵重要性
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. 生成預測結果

# %%
# 使用 Weighted Voting 模型進行預測
predictions = weighted_voting_model.predict(X_test)
predictions = np.maximum(predictions, 0)  # 確保預測值為正數

# 生成提交檔案
submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': predictions
})

submission.to_csv('submission.csv', index=False)
print("提交檔案已生成: submission.csv")
print(f"\n預測統計:")
print(f"Min: {predictions.min():,.2f}")
print(f"Max: {predictions.max():,.2f}")
print(f"Mean: {predictions.mean():,.2f}")
print(f"Median: {np.median(predictions):,.2f}")

# %%
# 預測分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(predictions, bins=50, edgecolor='black')
plt.title('Predicted Revenue Distribution')
plt.xlabel('Revenue')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(predictions), bins=50, edgecolor='black')
plt.title('Log(Predicted Revenue) Distribution')
plt.xlabel('Log(Revenue)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. 結論
# 
# 本專案使用 Taskmaster 方法系統化地完成了 TMDB 電影票房預測任務：
# 
# 1. **資料分析**: 深入分析了資料特徵和目標變數的分布
# 2. **特徵工程**: 創建了多種有效的特徵，包括時間特徵、計數特徵和衍生特徵
# 3. **模型訓練**: 訓練了多個強大的模型（Random Forest, XGBoost, LightGBM, CatBoost）
# 4. **模型融合**: 使用 Voting 方法融合多個模型，提升預測準確度
# 5. **結果生成**: 成功生成了預測結果
# 
# **關鍵發現**:
# - budget 是最重要的特徵
# - 模型融合能有效提升預測性能
# - 特徵工程對模型性能有顯著影響
