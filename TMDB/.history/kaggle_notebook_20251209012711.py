"""
TMDB Box Office Prediction - Kaggle Notebook Version
Clean and tested version for Kaggle environment
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import advanced models
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False

# Kaggle paths
TRAIN_PATH = '/kaggle/input/tmdb-box-office-prediction/train.csv'
TEST_PATH = '/kaggle/input/tmdb-box-office-prediction/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("TMDB Box Office Prediction - Kaggle Version")
print("=" * 70)

# ============================================================================
# Helper Functions
# ============================================================================

def parse_json_count(x):
    """Parse JSON and return count"""
    try:
        if pd.isna(x) or x == '':
            return 0
        data = json.loads(x.replace("'", '"'))
        return len(data)
    except:
        return 0

def parse_json_first_name(x):
    """Parse JSON and return first name"""
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
    """Feature engineering pipeline"""
    df = df.copy()
    
    # Numerical features
    df['budget'] = df['budget'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # Time features
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
    df['release_day'] = df['release_date'].dt.day.fillna(15).astype(int)
    df['release_dayofweek'] = df['release_date'].dt.dayofweek.fillna(4).astype(int)
    
    # JSON features
    df['genres_count'] = df['genres'].apply(parse_json_count)
    df['top_genre'] = df['genres'].apply(parse_json_first_name)
    df['cast_count'] = df['cast'].apply(parse_json_count)
    df['crew_count'] = df['crew'].apply(parse_json_count)
    df['keywords_count'] = df['Keywords'].apply(parse_json_count)
    df['production_companies_count'] = df['production_companies'].apply(parse_json_count)
    df['production_countries_count'] = df['production_countries'].apply(parse_json_count)
    df['spoken_languages_count'] = df['spoken_languages'].apply(parse_json_count)
    
    # Boolean features
    df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
    df['has_homepage'] = df['homepage'].notna().astype(int)
    df['has_tagline'] = df['tagline'].notna().astype(int)
    
    # Text length features
    df['title_length'] = df['title'].fillna('').apply(len)
    df['overview_length'] = df['overview'].fillna('').apply(len)
    df['tagline_length'] = df['tagline'].fillna('').apply(len)
    df['original_title_length'] = df['original_title'].fillna('').apply(len)
    
    # Derived features
    df['budget_popularity_ratio'] = df['budget'] / (df['popularity'] + 1)
    df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
    df['popularity_per_cast'] = df['popularity'] / (df['cast_count'] + 1)
    df['budget_per_company'] = df['budget'] / (df['production_companies_count'] + 1)
    
    # Log features
    df['log_budget'] = np.log1p(df['budget'])
    df['log_popularity'] = np.log1p(df['popularity'])
    
    # Categorical encoding
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
# Phase 1: Load Data
# ============================================================================
print("\n[Phase 1] Loading Data")
print("-" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train set: {train_df.shape}")
print(f"Test set: {test_df.shape}")

# ============================================================================
# Phase 2: Feature Engineering
# ============================================================================
print("\n[Phase 2] Feature Engineering")
print("-" * 70)

train_processed, label_encoders = feature_engineering(train_df, is_train=True)
test_processed, _ = feature_engineering(test_df, label_encoders=label_encoders, is_train=False)

print("Feature engineering completed")

# ============================================================================
# Phase 3: Prepare Training Data
# ============================================================================
print("\n[Phase 3] Preparing Training Data")
print("-" * 70)

feature_cols = [
    'budget', 'popularity', 'runtime',
    'release_year', 'release_month', 'release_quarter', 'release_day', 'release_dayofweek',
    'genres_count', 'cast_count', 'crew_count', 'keywords_count',
    'production_companies_count', 'production_countries_count', 'spoken_languages_count',
    'has_collection', 'has_homepage', 'has_tagline',
    'title_length', 'overview_length', 'tagline_length', 'original_title_length',
    'budget_popularity_ratio', 'budget_per_minute', 'popularity_per_cast', 'budget_per_company',
    'log_budget', 'log_popularity',
    'original_language', 'status', 'top_genre'
]

X = train_processed[feature_cols]
y = train_processed['revenue']
X_test = test_processed[feature_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {len(feature_cols)}")

# ============================================================================
# Phase 4: Model Training
# ============================================================================
print("\n[Phase 4] Model Training")
print("-" * 70)

models = {}

# Random Forest
print("\n[1/4] Training Random Forest...")
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
print(f"Random Forest - RMSE: ${rmse_rf:,.2f}, R2: {r2_rf:.4f}")

# XGBoost
if HAS_XGB:
    print("\n[2/4] Training XGBoost...")
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
    print(f"XGBoost - RMSE: ${rmse_xgb:,.2f}, R2: {r2_xgb:.4f}")
else:
    print("\n[2/4] Skipping XGBoost (not installed)")

# LightGBM
if HAS_LGB:
    print("\n[3/4] Training LightGBM...")
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
    print(f"LightGBM - RMSE: ${rmse_lgb:,.2f}, R2: {r2_lgb:.4f}")
else:
    print("\n[3/4] Skipping LightGBM (not installed)")

# CatBoost
if HAS_CAT:
    print("\n[4/4] Training CatBoost...")
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
    print(f"CatBoost - RMSE: ${rmse_cat:,.2f}, R2: {r2_cat:.4f}")
else:
    print("\n[4/4] Skipping CatBoost (not installed)")

# ============================================================================
# Phase 5: Model Ensemble
# ============================================================================
print("\n[Phase 5] Model Ensemble")
print("-" * 70)

if len(models) > 1:
    print(f"\nUsing {len(models)} models for Voting Ensemble...")
    
    estimators = [(name, model) for name, model in models.items()]
    voting_model = VotingRegressor(estimators)
    voting_model.fit(X_train, y_train)
    
    y_pred_voting = voting_model.predict(X_val)
    rmse_voting = np.sqrt(mean_squared_error(y_val, y_pred_voting))
    r2_voting = r2_score(y_val, y_pred_voting)
    print(f"Voting Ensemble - RMSE: ${rmse_voting:,.2f}, R2: {r2_voting:.4f}")
    
    final_model = voting_model
    final_model_name = "Voting Ensemble"
else:
    print("\nUsing Random Forest as final model")
    final_model = rf_model
    final_model_name = "Random Forest"

# ============================================================================
# Phase 6: Feature Importance
# ============================================================================
print("\n[Phase 6] Feature Importance Analysis")
print("-" * 70)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# Phase 7: Generate Predictions
# ============================================================================
print("\n[Phase 7] Generating Predictions")
print("-" * 70)

predictions = final_model.predict(X_test)
predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': predictions
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission file saved: {OUTPUT_PATH}")

print(f"\nPrediction Statistics:")
print(f"  Min: ${predictions.min():,.2f}")
print(f"  Max: ${predictions.max():,.2f}")
print(f"  Mean: ${predictions.mean():,.2f}")
print(f"  Median: ${np.median(predictions):,.2f}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 70)
print("Project Completed Successfully")
print("=" * 70)

print(f"\n[Final Model] {final_model_name}")
print(f"\n[Validation Performance]")
y_pred_final = final_model.predict(X_val)
rmse_final = np.sqrt(mean_squared_error(y_val, y_pred_final))
mae_final = mean_absolute_error(y_val, y_pred_final)
r2_final = r2_score(y_val, y_pred_final)

print(f"  RMSE: ${rmse_final:,.2f}")
print(f"  MAE:  ${mae_final:,.2f}")
print(f"  R2:   {r2_final:.4f}")

print(f"\n[Models Used]")
for name in models.keys():
    print(f"  - {name.upper()}")

print(f"\n[Number of Features] {len(feature_cols)}")

print(f"\n[Output File]")
print(f"  {OUTPUT_PATH}")

print("\n" + "=" * 70)
print("Ready to submit to Kaggle!")
print("=" * 70)

print("\n[Submission Preview]")
print(submission.head(10))
