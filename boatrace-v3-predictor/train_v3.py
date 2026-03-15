import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
from tabpfn import TabPFNClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import warnings

# --- TabPFN / PyTorch 2.6+ Compatibility Fix ---
# Python 3.13 / PyTorch 2.6+ restrict torch.load for security.
# TabPFN 0.1.9 relies on older serialization. Let's patch it.
original_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_torch_load
# -----------------------------------------------

# --- Configuration ---
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# GPU Settings (AMD ROCm / General GPU)
# LightGBM CPU for stability with high-cardinality features on some ROCm versions
LGB_PARAMS_BASE = {
    'device': 'cpu',
    'verbosity': -1
}

CB_PARAMS_BASE = {
    'task_type': 'CPU', # Switched to CPU for max compatibility during debug
    'random_seed': 42,
    'verbose': 100
}

def load_data(limit=20000):
    """データベースから学習データをロードし、前回と同様の特徴量エンジニアリングを施す"""
    print(f"Loading latest {limit} rows from boatrace.db...")
    conn = sqlite3.connect(DB_PATH)
    
    # 既存の make_data_set.py のクエリをベースにする
    query = f"""
    SELECT
        re.race_id, re.boat_number, re.racer_id, r.venue_code,
        bi.exhibition_time, bi.exhibition_start_timing, 
        COALESCE(bi.exhibition_entry_course, re.boat_number) as pred_course,
        re.nat_win_rate, re.motor_rate, re.boat_rate, re.prior_results,
        res.finish_order as rank
    FROM race_entries re
    JOIN races r ON re.race_id = r.race_id
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    LEFT JOIN results res ON re.race_id = res.race_id AND re.boat_number = res.boat_number
    WHERE res.finish_order IS NOT NULL
    ORDER BY r.race_date DESC, re.race_id DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 基本的なデータ型変換と欠損処理を行ってから categorical にする
    df['target'] = (df['rank'] == 1).astype(int)
    # racer_id 等、不要にメモリを食わないよう注意
    df['racer_id'] = df['racer_id'].fillna(0).astype(int).astype(str).astype('category')
    df['venue_code'] = df['venue_code'].fillna(0).astype(int).astype(str).astype('category')
    
    # 数値列の NaN を埋める
    num_cols = ['exhibition_time', 'exhibition_start_timing', 'nat_win_rate', 'motor_rate', 'boat_rate']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not df[col].isna().all() else 0)
    
    return df

def train_lgb(X_train, y_train, X_val, y_val, cat_features):
    print("\n--- Training LightGBM with Optuna (CPU Mode) ---")
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            **LGB_PARAMS_BASE
        }
        
        train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, categorical_feature=cat_features)
        
        gbm = lgb.train(param, train_ds, valid_sets=[val_ds], 
                        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)])
        preds = gbm.predict(X_val)
        return log_loss(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) # Further reduced for speed
    
    best_params = study.best_params
    best_params.update(LGB_PARAMS_BASE)
    best_params['objective'] = 'binary'
    
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, categorical_feature=cat_features)
    
    model = lgb.train(best_params, train_ds, valid_sets=[val_ds],
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])
    return model

def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    print("\n--- Training CatBoost (CPU Mode) ---")
    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=50,
        **CB_PARAMS_BASE
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    return model

def train_alt_model(X_train, y_train, X_val, y_val):
    """TabPFN の外部依存問題を回避するための代替基底モデル (Strong Regularized LR)"""
    print("\n--- Training Alternative Model (L2-Regularized LR) ---")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # 欠損値補完は load_data で済んでいる前提、スケーリングを行う
    scaler = StandardScaler()
    
    # カテゴリ変数をダミー変数化（高カーディナリティの racer_id は除外して安定化）
    X_train_sub = X_train.drop(X_train.select_dtypes(['category']).columns, axis=1)
    X_val_sub = X_val.drop(X_val.select_dtypes(['category']).columns, axis=1)
    
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val_sub)
    
    # 強力な正則化を施した LR
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, X_val_scaled, scaler

def main():
    df = load_data(30000)
    
    features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_id']
    cat_features = ['venue_code', 'racer_id']
    target = 'target'
    
    # Train/Val Split
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    
    # 1. LightGBM
    model_lgb = train_lgb(X_train, y_train, X_val, y_val, cat_features)
    lgb_val_preds = model_lgb.predict(X_val)
    print(f"LGB LogLoss: {log_loss(y_val, lgb_val_preds):.4f}")
    
    # 2. CatBoost
    model_cb = train_catboost(X_train, y_train, X_val, y_val, cat_features)
    cb_val_preds = model_cb.predict_proba(X_val)[:, 1]
    print(f"CB LogLoss: {log_loss(y_val, cb_val_preds):.4f}")
    
    # 3. Alternative Model (แทน TabPFN)
    model_alt, X_val_alt, alt_scaler = train_alt_model(X_train, y_train, X_val, y_val)
    alt_val_preds = model_alt.predict_proba(X_val_alt)[:, 1]
    print(f"Alt-Model LogLoss: {log_loss(y_val, alt_val_preds):.4f}")

    # --- Stacking ---
    print("\n--- Meta-Stacking (V3.1: LGBM+CB+AltModel) ---")
    X_meta = np.vstack([lgb_val_preds, cb_val_preds, alt_val_preds]).T
    
    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_val)
    
    final_preds = meta_model.predict_proba(X_meta)[:, 1]
    print(f"Stacked Model (V3.1) LogLoss: {log_loss(y_val, final_preds):.4f}")
    print(f"Stacked Model (V3.1) Accuracy: {accuracy_score(y_val, (final_preds > 0.5).astype(int)):.4f}")
    
    # Save Models
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    model_lgb.save_model(os.path.join(MODEL_DIR, 'lgb_model.txt'))
    model_cb.save_model(os.path.join(MODEL_DIR, 'cb_model.bin'))
    with open(os.path.join(MODEL_DIR, 'alt_model.pkl'), 'wb') as f:
        pickle.dump(model_alt, f)
    with open(os.path.join(MODEL_DIR, 'alt_scaler.pkl'), 'wb') as f:
        pickle.dump(alt_scaler, f)
    # Note: TabPFN itself is not easily pickled, will rely on retraining/inference if needed or fresh fit in App
    # In V3.1 we'll attempt to save TabPFN if possible, or handle it in App.
    
    # Visualization: Feature Importance (LGBM and CatBoost)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lgb.plot_importance(model_lgb, ax=axes[0], title='LGBM Importance')
    cb_imp = pd.Series(model_cb.get_feature_importance(), index=features).sort_values()
    cb_imp.plot(kind='barh', ax=axes[1], title='CatBoost Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance saved to feature_importance.png")

if __name__ == "__main__":
    main()
