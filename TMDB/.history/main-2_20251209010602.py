"""
TMDB 電影票房預測 - Kaggle Notebook 版本
適用於 Kaggle 環境的完整流程
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 嘗試導入進階模型（如果可用）
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠ XGBoost 未安裝")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("⚠ LightGBM 未安裝")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False
    print("⚠ CatBoost 未安裝")

# Kaggle 路徑設定
TRAIN_PATH = '/kaggle/input/tmdb-box-office-prediction/train.csv'
TEST_PATH = '/kaggle/input/tmdb-box-office-prediction/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print(" " * 15 + "TMDB 電影票房預測 - Kaggle 版本")
print("=" * 70)

# ============================================================================
# 工具函數
# ============================================================================

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
    
    # 1. 數值特徵處理
    df['budget'] = df['budget'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # 2. 時間特徵
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
    df['release_day'] = df['release_date'].dt.day.fillna(15).astype(int)
    df['release_dayofweek'] = df['release_date'].dt.dayofweek.fillna(4).astype(int)
    
    # 3. JSON 欄位處理
    df['genres_count'] = df['genres'].apply(parse_json_count)
    df['top_genre'] = df['genres'].apply(parse_json_first_name)
    df['cast_count'] = df['cast'].apply(parse_json_count)
    df['crew_count'] = df['crew'].apply(parse_json_count)
    df['keywords_count'] = df['Keywords'].apply(parse_json_count)
    df['production_companies_count'] = df['production_companies'].apply(parse_json_count)
    df['production_countries_count'] = df['production_countries'].apply(parse_json_count)
    df['spoken_languages_count'] = df['spoken_languages'].apply(parse_json_count)
    
    # 4. 布林特徵
    df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
    df['has_homepage'] = df['homepage'].notna().astype(int)
    df['has_tagline'] = df['tagline'].notna().astype(int)
    
    # 5. 文本長度特徵
    df['title_length'] = df['title'].fillna('').apply(len)
    df['overview_length'] = df['overview'].fillna('').apply(len)
    df['tagline_length'] = df['tagline'].fillna('').apply(len)
    df['original_title_length'] = df['original_title'].fillna('').apply(len)
    
    # 6. 衍生特徵
    df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)
    df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
    df['popularity_per_cast'] = df['popularity'] / (df['cast_count'] + 1)
    df['budget_per_company'] = df['budget'] / (df['production_companies_count'] + 1)
    
    # 7. 對數轉換（處理偏態）
    df['log_budget'] = np.log1p(df['budget'])
    df['log_popularity'] = np.log1p(df['popularity'])
    
    # 8. 類別特徵編碼
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
            def safe_transform(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                else:
                    return le.transform([le.classes_[0]])[0]
            df[col] = df[col].astype(str).apply(safe_transform)
    
    return df, label_encoders

# ============================================================================
# Phase 1: 載入資料
# ============================================================================
print("\n【Phase 1】載入資料")
print("-" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"✓ 訓練集: {train_df.shape}")
print(f"✓ 測試集: {test_df.shape}")

# ============================================================================
# Phase 2: 特徵工程
# ============================================================================
print("\n【Phase 2】特徵工程")
print("-" * 70)

train_processed, label_encoders = feature_engineering(train_df, is_train=True)
test_processed, _ = feature_engineering(test_df, label_encoders=label_encoders, is_train=False)

print(f"✓ 特徵工程完成")

# ============================================================================
# Phase 3: 準備訓練資料
# ============================================================================
print("\n【Phase 3】準備訓練資料")
print("-" * 70)

feature_cols = [
    # 原始數值特徵
    'budget', 'popularity', 'runtime',
    # 時間特徵
    'release_year', 'release_month', 'release_quarter', 'release_day', 'release_dayofweek',
    # 計數特徵
    'genres_count', 'cast_count', 'crew_count', 'keywords_count',
    'production_companies_count', 'production_countries_count', 'spoken_languages_count',
    # 布林特徵
    'has_collection', 'has_homepage', 'has_tagline',
    # 文本長度特徵
    'title_length', 'overview_length', 'tagline_length', 'original_title_length',
    # 衍生特徵
    'budget_popularity_ratio', 'budget_per_minute', 'popularity_per_cast', 'budget_per_company',
    # 對數特徵
    'log_budget', 'log_popularity',
    # 類別特徵
    'original_language', 'status', 'top_genre'
]

X = train_processed[feature_cols]
y = train_processed['revenue']
X_test = test_processed[feature_cols]

# 分割訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"✓ 訓練集: {X_train.shape}")
print(f"✓ 驗證集: {X_val.shape}")
print(f"✓ 測試集: {X_test.shape}")
print(f"✓ 特徵數量: {len(feature_cols)}")

# ============================================================================
# Phase 4: 模型訓練
# ============================================================================
print("\n【Phase 4】模型訓練")
print("-" * 70)

models = {}

# 1. Random Forest
print("\n[1/4] 訓練 Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
models['rf'] = rf_model

y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf = r2_score(y_val, y_pred_rf)
print(f"✓ Random Forest - RMSE: ${rmse_rf:,.2f}, R²: {r2_rf:.4f}")

# 2. XGBoost
if HAS_XGB:
    print("\n[2/4] 訓練 XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    
    y_pred_xgb = xgb_model.predict(X_val)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
    r2_xgb = r2_score(y_val, y_pred_xgb)
    print(f"✓ XGBoost - RMSE: ${rmse_xgb:,.2f}, R²: {r2_xgb:.4f}")
else:
    print("\n[2/4] ⚠ 跳過 XGBoost (未安裝)")

# 3. LightGBM
if HAS_LGB:
    print("\n[3/4] 訓練 LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    
    y_pred_lgb = lgb_model.predict(X_val)
    rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
    r2_lgb = r2_score(y_val, y_pred_lgb)
    print(f"✓ LightGBM - RMSE: ${rmse_lgb:,.2f}, R²: {r2_lgb:.4f}")
else:
    print("\n[3/4] ⚠ 跳過 LightGBM (未安裝)")

# 4. CatBoost
if HAS_CAT:
    print("\n[4/4] 訓練 CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=200,
        depth=8,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['cat'] = cat_model
    
    y_pred_cat = cat_model.predict(X_val)
    rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))
    r2_cat = r2_score(y_val, y_pred_cat)
    print(f"✓ CatBoost - RMSE: ${rmse_cat:,.2f}, R²: {r2_cat:.4f}")
else:
    print("\n[4/4] ⚠ 跳過 CatBoost (未安裝)")

# ============================================================================
# Phase 5: 模型融合
# ============================================================================
print("\n【Phase 5】模型融合")
print("-" * 70)

if len(models) > 1:
    print(f"\n使用 {len(models)} 個模型進行 Voting Ensemble...")
    
    # 準備 Voting Regressor
    estimators = [(name, model) for name, model in models.items()]
    
    # Simple Voting
    voting_model = VotingRegressor(estimators)
    voting_model.fit(X_train, y_train)
    
    y_pred_voting = voting_model.predict(X_val)
    rmse_voting = np.sqrt(mean_squared_error(y_val, y_pred_voting))
    r2_voting = r2_score(y_val, y_pred_voting)
    print(f"✓ Voting Ensemble - RMSE: ${rmse_voting:,.2f}, R²: {r2_voting:.4f}")
    
    # 使用 Voting 作為最終模型
    final_model = voting_model
    final_model_name = "Voting Ensemble"
else:
    print("\n只有一個模型，使用 Random Forest 作為最終模型")
    final_model = rf_model
    final_model_name = "Random Forest"

# ============================================================================
# Phase 6: 特徵重要性分析
# ============================================================================
print("\n【Phase 6】特徵重要性分析")
print("-" * 70)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 重要特徵:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# Phase 7: 生成預測
# ============================================================================
print("\n【Phase 7】生成預測結果")
print("-" * 70)

predictions = final_model.predict(X_test)
predictions = np.maximum(predictions, 0)  # 確保預測值為正數

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': predictions
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"✓ 提交檔案已生成: {OUTPUT_PATH}")

print(f"\n預測統計:")
print(f"  最小值: ${predictions.min():,.2f}")
print(f"  最大值: ${predictions.max():,.2f}")
print(f"  平均值: ${predictions.mean():,.2f}")
print(f"  中位數: ${np.median(predictions):,.2f}")

# ============================================================================
# 最終總結
# ============================================================================
print("\n" + "=" * 70)
print(" " * 25 + "專案完成總結")
print("=" * 70)

print(f"\n【最終模型】{final_model_name}")
print(f"\n【驗證集性能】")
y_pred_final = final_model.predict(X_val)
rmse_final = np.sqrt(mean_squared_error(y_val, y_pred_final))
mae_final = mean_absolute_error(y_val, y_pred_final)
r2_final = r2_score(y_val, y_pred_final)

print(f"  RMSE: ${rmse_final:,.2f}")
print(f"  MAE:  ${mae_final:,.2f}")
print(f"  R²:   {r2_final:.4f}")

print(f"\n【使用的模型】")
for name in models.keys():
    print(f"  ✓ {name.upper()}")

print(f"\n【特徵數量】{len(feature_cols)}")

print(f"\n【輸出檔案】")
print(f"  ✓ {OUTPUT_PATH}")

print("\n" + "=" * 70)
print(" " * 20 + "預測完成，可以提交到 Kaggle！")
print("=" * 70)

# 顯示前幾筆預測結果
print("\n【預測結果預覽】")
print(submission.head(10))
