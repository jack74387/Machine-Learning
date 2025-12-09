"""
TMDB 電影票房預測 - 主程式
使用 Taskmaster 方法進行系統化開發
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# 設定隨機種子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class TMDBPredictor:
    """TMDB 電影票房預測器"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """載入資料"""
        print("=" * 50)
        print("載入資料...")
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        print(f"訓練集大小: {self.train_df.shape}")
        print(f"測試集大小: {self.test_df.shape}")
        return self
    
    def explore_data(self):
        """探索性資料分析"""
        print("\n" + "=" * 50)
        print("探索性資料分析...")
        
        # 基本資訊
        print("\n訓練集資訊:")
        print(self.train_df.info())
        
        # 目標變數統計
        print("\n目標變數 (revenue) 統計:")
        print(self.train_df['revenue'].describe())
        
        # 缺失值統計
        print("\n缺失值統計:")
        missing = self.train_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print(missing)
        
        return self
    
    def parse_json_column(self, df, column, key=None, count=False):
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
    
    def feature_engineering(self, df, is_train=True):
        """特徵工程"""
        print("\n" + "=" * 50)
        print("特徵工程...")
        
        df = df.copy()
        
        # 1. 處理基本數值特徵
        df['budget'] = df['budget'].fillna(0)
        df['popularity'] = df['popularity'].fillna(0)
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        
        # 2. 處理時間特徵
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
        df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
        df['release_day'] = df['release_date'].dt.day.fillna(15).astype(int)
        df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
        
        # 3. 處理 JSON 欄位
        # genres
        df['genres_count'] = self.parse_json_column(df, 'genres', count=True)
        df['top_genre'] = self.parse_json_column(df, 'genres', key='name')
        
        # cast
        df['cast_count'] = self.parse_json_column(df, 'cast', count=True)
        
        # crew
        df['crew_count'] = self.parse_json_column(df, 'crew', count=True)
        
        # keywords
        df['keywords_count'] = self.parse_json_column(df, 'Keywords', count=True)
        
        # production_companies
        df['production_companies_count'] = self.parse_json_column(df, 'production_companies', count=True)
        
        # 4. 是否屬於系列電影
        df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
        
        # 5. 文本長度特徵
        df['title_length'] = df['title'].fillna('').apply(len)
        df['overview_length'] = df['overview'].fillna('').apply(len)
        df['tagline_length'] = df['tagline'].fillna('').apply(len)
        
        # 6. 衍生特徵
        df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)
        df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
        
        # 7. 處理類別特徵
        df['original_language'] = df['original_language'].fillna('en')
        df['status'] = df['status'].fillna('Released')
        df['top_genre'] = df['top_genre'].fillna('Unknown')
        
        # Label Encoding
        categorical_features = ['original_language', 'status', 'top_genre']
        for col in categorical_features:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # 處理測試集中的新類別
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                df[col] = le.transform(df[col].astype(str))
        
        print(f"特徵工程完成，特徵數量: {df.shape[1]}")
        return df
    
    def prepare_features(self):
        """準備訓練特徵"""
        print("\n" + "=" * 50)
        print("準備訓練特徵...")
        
        # 特徵工程
        self.train_processed = self.feature_engineering(self.train_df, is_train=True)
        self.test_processed = self.feature_engineering(self.test_df, is_train=False)
        
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
        
        self.X = self.train_processed[feature_cols]
        self.y = self.train_processed['revenue']
        self.X_test = self.test_processed[feature_cols]
        
        # 分割訓練集和驗證集
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        print(f"訓練集大小: {self.X_train.shape}")
        print(f"驗證集大小: {self.X_val.shape}")
        print(f"測試集大小: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """訓練多個模型"""
        print("\n" + "=" * 50)
        print("訓練模型...")
        
        # 1. Random Forest
        print("\n訓練 Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['rf'] = rf_model
        self._evaluate_model('Random Forest', rf_model)
        
        # 2. XGBoost
        print("\n訓練 XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgb'] = xgb_model
        self._evaluate_model('XGBoost', xgb_model)
        
        # 3. LightGBM
        print("\n訓練 LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(self.X_train, self.y_train)
        self.models['lgb'] = lgb_model
        self._evaluate_model('LightGBM', lgb_model)
        
        # 4. CatBoost
        print("\n訓練 CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=False
        )
        cat_model.fit(self.X_train, self.y_train)
        self.models['cat'] = cat_model
        self._evaluate_model('CatBoost', cat_model)
        
        return self
    
    def _evaluate_model(self, name, model):
        """評估單一模型"""
        y_pred = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        mae = mean_absolute_error(self.y_val, y_pred)
        r2 = r2_score(self.y_val, y_pred)
        
        print(f"{name} - RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R²: {r2:.4f}")
    
    def ensemble_voting(self):
        """模型融合 - Voting"""
        print("\n" + "=" * 50)
        print("模型融合 - Voting Regressor...")
        
        # Simple Voting
        voting_model = VotingRegressor([
            ('rf', self.models['rf']),
            ('xgb', self.models['xgb']),
            ('lgb', self.models['lgb']),
            ('cat', self.models['cat'])
        ])
        voting_model.fit(self.X_train, self.y_train)
        self.models['voting'] = voting_model
        self._evaluate_model('Voting (Simple)', voting_model)
        
        # Weighted Voting (根據驗證集表現調整權重)
        weights = [0.2, 0.3, 0.3, 0.2]  # 可根據實際表現調整
        weighted_voting_model = VotingRegressor([
            ('rf', self.models['rf']),
            ('xgb', self.models['xgb']),
            ('lgb', self.models['lgb']),
            ('cat', self.models['cat'])
        ], weights=weights)
        weighted_voting_model.fit(self.X_train, self.y_train)
        self.models['weighted_voting'] = weighted_voting_model
        self._evaluate_model('Voting (Weighted)', weighted_voting_model)
        
        return self
    
    def predict_and_submit(self, model_name='weighted_voting'):
        """預測並生成提交檔案"""
        print("\n" + "=" * 50)
        print(f"使用 {model_name} 模型進行預測...")
        
        model = self.models[model_name]
        predictions = model.predict(self.X_test)
        
        # 確保預測值為正數
        predictions = np.maximum(predictions, 0)
        
        # 生成提交檔案
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'revenue': predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        print("提交檔案已生成: submission.csv")
        print(f"預測統計: min={predictions.min():,.2f}, max={predictions.max():,.2f}, mean={predictions.mean():,.2f}")
        
        return submission
    
    def run_pipeline(self):
        """執行完整流程"""
        self.load_data()
        self.explore_data()
        self.prepare_features()
        self.train_models()
        self.ensemble_voting()
        submission = self.predict_and_submit()
        
        print("\n" + "=" * 50)
        print("流程完成！")
        return submission


if __name__ == "__main__":
    predictor = TMDBPredictor()
    submission = predictor.run_pipeline()
