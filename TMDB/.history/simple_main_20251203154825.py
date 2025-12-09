"""
TMDB 電影票房預測 - 簡化版
只使用基本套件進行演示
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("TMDB 電影票房預測專案 - 簡化版")
print("=" * 60)

# 1. 載入資料
print("\n[1/6] 載入資料...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"✓ 訓練集大小: {train_df.shape}")
print(f"✓ 測試集大小: {test_df.shape}")

# 2. 基本資料探索
print("\n[2/6] 資料探索...")
print(f"✓ 目標變數 (revenue) 統計:")
print(f"  - 平均值: ${train_df['revenue'].mean():,.0f}")
print(f"  - 中位數: ${train_df['revenue'].median():,.0f}")
print(f"  - 最大值: ${train_df['revenue'].max():,.0f}")
print(f"  - 最小值: ${train_df['revenue'].min():,.0f}")

# 3. 特徵工程
print("\n[3/6] 特徵工程...")

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
    
    # 數值特徵
    df['budget'] = df['budget'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # 時間特徵
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
    
    # JSON 欄位
    df['genres_count'] = df['genres'].apply(parse_json_count)
    df['top_genre'] = df['genres'].apply(parse_json_first_name)
    df['cast_count'] = df['cast'].apply(parse_json_count)
    df['crew_count'] = df['crew'].apply(parse_json_count)
    
    # 系列電影
    df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
    
    # 文本長度
    df['title_length'] = df['title'].fillna('').apply(len)
    df['overview_length'] = df['overview'].fillna('').apply(len)
    
    # 衍生特徵
    df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)
    
    # 類別特徵編碼
    df['original_language'] = df['original_language'].fillna('en')
    df['top_genre'] = df['top_genre'].fillna('Unknown')
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in ['original_language', 'top_genre']:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # 處理測試集中的新類別，將其映射到訓練集中最常見的類別
            def safe_transform(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                else:
                    return le.transform([le.classes_[0]])[0]  # 使用第一個類別作為預設值
            df[col] = df[col].astype(str).apply(safe_transform)
    
    return df, label_encoders

train_processed, label_encoders = feature_engineering(train_df, is_train=True)
test_processed, _ = feature_engineering(test_df, label_encoders=label_encoders, is_train=False)
print(f"✓ 特徵工程完成")

# 4. 準備訓練資料
print("\n[4/6] 準備訓練資料...")
feature_cols = [
    'budget', 'popularity', 'runtime',
    'release_year', 'release_month',
    'genres_count', 'cast_count', 'crew_count',
    'has_collection', 'title_length', 'overview_length',
    'budget_popularity_ratio',
    'original_language', 'top_genre'
]

X = train_processed[feature_cols]
y = train_processed['revenue']
X_test = test_processed[feature_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"✓ 訓練集: {X_train.shape}")
print(f"✓ 驗證集: {X_val.shape}")

# 5. 訓練模型
print("\n[5/6] 訓練模型...")
print("使用 Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("✓ 模型訓練完成")

# 評估
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\n模型性能:")
print(f"  - RMSE: ${rmse:,.2f}")
print(f"  - MAE: ${mae:,.2f}")
print(f"  - R²: {r2:.4f}")

# 特徵重要性
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 重要特徵:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# 6. 生成預測
print("\n[6/6] 生成預測結果...")
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': predictions
})

submission.to_csv('submission.csv', index=False)
print("✓ 提交檔案已生成: submission.csv")
print(f"\n預測統計:")
print(f"  - 最小值: ${predictions.min():,.2f}")
print(f"  - 最大值: ${predictions.max():,.2f}")
print(f"  - 平均值: ${predictions.mean():,.2f}")
print(f"  - 中位數: ${np.median(predictions):,.2f}")

print("\n" + "=" * 60)
print("專案完成！")
print("=" * 60)
print("\n下一步:")
print("1. 查看 task.md 了解完整的任務規劃")
print("2. 查看 report.md 了解詳細的專案報告")
print("3. 執行 install_packages.bat 安裝完整套件")
print("4. 執行 main.py 使用完整版本（包含多模型融合）")
