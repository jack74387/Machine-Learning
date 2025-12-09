"""
訓練終極模型的函數
"""

def train_models(self):
    """訓練終極模型組合"""
    print("\n[4/8] 訓練終極模型組合...")
    
    # 1. Random Forest（優化參數）
    print("\n[1/6] 訓練 Random Forest...")
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(self.X_train, self.y_train)
    self.models['rf'] = rf_model
    self._evaluate_model('Random Forest', rf_model)
    
    # 2. Gradient Boosting（優化參數）
    print("\n[2/6] 訓練 Gradient Boosting...")
    from sklearn.ensemble import GradientBoostingRegressor
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        verbose=0
    )
    gb_model.fit(self.X_train, self.y_train)
    self.models['gb'] = gb_model
    self._evaluate_model('Gradient Boosting', gb_model)
    
    # 3. XGBoost（如果可用）
    if HAS_XGB:
        print("\n[3/6] 訓練 XGBoost...")
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgb'] = xgb_model
        self._evaluate_model('XGBoost', xgb_model)
    
    # 4. LightGBM（如果可用）
    if HAS_LGB:
        print("\n[4/6] 訓練 LightGBM...")
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(self.X_train, self.y_train)
        self.models['lgb'] = lgb_model
        self._evaluate_model('LightGBM', lgb_model)
    
    # 5. CatBoost（如果可用）
    if HAS_CAT:
        print("\n[5/6] 訓練 CatBoost...")
        from catboost import CatBoostRegressor
        cat_model = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.02,
            l2_leaf_reg=3,
            random_state=RANDOM_STATE,
            verbose=False
        )
        cat_model.fit(self.X_train, self.y_train)
        self.models['cat'] = cat_model
        self._evaluate_model('CatBoost', cat_model)
    
    # 6. Ridge（作為基準）
    print("\n[6/6] 訓練 Ridge...")
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=20.0, random_state=RANDOM_STATE)
    ridge_model.fit(self.X_train, self.y_train)
    self.models['ridge'] = ridge_model
    self._evaluate_model('Ridge', ridge_model)
    
    return self


def ensemble_predict(self):
    """終極集成預測"""
    print("\n[5/8] 終極集成預測...")
    
    # 動態權重（根據可用模型）
    if HAS_XGB and HAS_LGB and HAS_CAT:
        weights = {
            'rf': 0.10,
            'gb': 0.15,
            'xgb': 0.30,
            'lgb': 0.30,
            'cat': 0.15
        }
        print("使用完整模型組合（5個模型）")
    elif HAS_XGB and HAS_LGB:
        weights = {
            'rf': 0.15,
            'gb': 0.20,
            'xgb': 0.35,
            'lgb': 0.30
        }
        print("使用 4 個模型組合")
    else:
        weights = {
            'rf': 0.40,
            'gb': 0.50,
            'ridge': 0.10
        }
        print("使用基礎模型組合（3個模型）")
    
    predictions = {}
    for name, model in self.models.items():
        if name in weights:
            pred = model.predict(self.X_test)
            if self.use_log_transform:
                pred = np.expm1(pred)
            predictions[name] = pred
    
    # 加權平均
    final_pred = sum(predictions[name] * weights[name] for name in weights.keys() if name in predictions)
    final_pred = np.maximum(final_pred, 0)
    
    # 評估集成模型
    print("\n評估集成模型...")
    ensemble_val_pred = sum(
        (np.expm1(model.predict(self.X_val)) if self.use_log_transform else model.predict(self.X_val)) * weights.get(name, 0)
        for name, model in self.models.items() if name in weights
    )
    y_val_original = np.expm1(self.y_val) if self.use_log_transform else self.y_val
    
    rmse = np.sqrt(mean_squared_error(y_val_original, ensemble_val_pred))
    mae = mean_absolute_error(y_val_original, ensemble_val_pred)
    r2 = r2_score(y_val_original, ensemble_val_pred)
    
    print(f"{'Ensemble':20s} - RMSE: ${rmse:,.0f}, MAE: ${mae:,.0f}, R²: {r2:.4f}")
    
    return final_pred
