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
ENABLE_L2_FEATURE_SELECTION = False # Experiment Flag: V3.1+ logic
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
    
    # --- Feature Engineering: Phase 1 Relative Features ---
    print("Adding Phase 1: Relative Features...")
    
    # 展示タイムの偏差値 (小さいほど良い)
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else 0

    df['exhibition_time_z'] = df.groupby('race_id')['exhibition_time'].transform(zscore)
    df['nat_win_rate_z'] = df.groupby('race_id')['nat_win_rate'].transform(zscore)
    
    # 1号艇との差分
    b1_win_rates = df[df['boat_number'] == 1][['race_id', 'nat_win_rate']].rename(columns={'nat_win_rate': 'b1_nat_win_rate'})
    df = df.merge(b1_win_rates, on='race_id', how='left')
    df['win_rate_diff_b1'] = df['nat_win_rate'] - df['b1_nat_win_rate'].fillna(df['nat_win_rate'])
    df.drop('b1_nat_win_rate', axis=1, inplace=True)
    
    # --- Standard Preprocessing ---
    # rank 1st -> 5, 2nd -> 4, ..., 6th -> 0 (for lambdarank relevance)
    df['rank_target'] = (6 - df['rank']).astype(int)
    df['target'] = (df['rank'] == 1).astype(int)
    
    # --- Feature Engineering: Phase 2 Target Encoding ---
    print("Adding Phase 2: K-Fold Target Encoding for racer_id...")
    from sklearn.model_selection import KFold
    df['racer_target_enc'] = 0.0
    kf = KFold(n_splits=5, shuffle=False)
    global_mean = df['target'].mean()
    df['racer_id_str'] = df['racer_id'].fillna(0).astype(int).astype(str)
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        racer_stats = train_fold.groupby('racer_id_str')['target'].agg(['mean', 'count'])
        m = 10
        racer_stats['smoothed'] = (racer_stats['count'] * racer_stats['mean'] + m * global_mean) / (racer_stats['count'] + m)
        mapping = racer_stats['smoothed'].to_dict()
        df.loc[df.index[val_idx], 'racer_target_enc'] = df.loc[df.index[val_idx], 'racer_id_str'].map(mapping).fillna(global_mean)

    df['racer_id'] = df['racer_id'].fillna(0).astype(int).astype(str).astype('category')
    df['venue_code'] = df['venue_code'].fillna(0).astype(int).astype(str).astype('category')
    
    num_cols = ['exhibition_time', 'exhibition_start_timing', 'nat_win_rate', 'motor_rate', 'boat_rate', 
                'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1', 'racer_target_enc']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Final Mapping for Inference (using ALL available data for the map) ---
    racer_stats = df.groupby('racer_id_str')['target'].agg(['mean', 'count'])
    # Smoothing m=10 (same as in K-Fold loop)
    m = 10
    racer_stats['smoothed'] = (racer_stats['count'] * racer_stats['mean'] + m * global_mean) / (racer_stats['count'] + m)
    racer_mapping = racer_stats['smoothed'].to_dict()
    racer_mapping['global_mean'] = global_mean

    return df, racer_mapping

def softmax_calibration(scores, group_sizes):
    probs = np.zeros_like(scores)
    idx = 0
    for size in group_sizes:
        if size == 0: continue
        s = scores[idx:idx+size]
        e_x = np.exp(s - np.max(s))
        probs[idx:idx+size] = e_x / e_x.sum()
        idx += size
    return probs

def train_lgb(X_train, y_train, X_val, y_val, train_groups, val_groups, cat_features):
    print("\n--- Training LightGBM LambdaRank (CPU Mode) ---")
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [1, 3],
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            **LGB_PARAMS_BASE
        }
        train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, categorical_feature=cat_features)
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, categorical_feature=cat_features)
        gbm = lgb.train(param, train_ds, valid_sets=[val_ds], 
                        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)])
        return gbm.best_score['valid_0']['ndcg@1']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    
    best_params = study.best_params
    best_params.update(LGB_PARAMS_BASE)
    best_params['objective'] = 'lambdarank'
    
    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, categorical_feature=cat_features)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, categorical_feature=cat_features)
    model = lgb.train(best_params, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(stopping_rounds=50)])
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
    print(f"\n--- Training Alternative Model (L2-Regularized LR) [Feature Selection: {ENABLE_L2_FEATURE_SELECTION}] ---")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # 欠損値補完は load_data で済んでいる前提、スケーリングを行う
    scaler = StandardScaler()
    
    if ENABLE_L2_FEATURE_SELECTION:
        # 改良版: 線形モデルに相性の良い、順序に意味のある数値変数のみを厳選
        selected_features = [
            'exhibition_time', 'exhibition_start_timing', 'pred_course',
            'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_target_enc'
        ]
        X_train_sub = X_train[selected_features]
        X_val_sub = X_val[selected_features]
        print(f"  Selected Features for Alt-L2: {selected_features}")
    else:
        # 従来版 (V3.1): カテゴリ変数以外をすべて使用
        X_train_sub = X_train.drop(X_train.select_dtypes(['category']).columns, axis=1)
        X_val_sub = X_val.drop(X_val.select_dtypes(['category']).columns, axis=1)
        print(f"  Legacy Mode: Using all numerical columns: {list(X_train_sub.columns)}")
    
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val_sub)
    
    # 強力な正則化を施した LR
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, X_val_scaled, scaler

def main():
    df, racer_mapping = load_data(30000)
    
    features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_id',
                'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1', 'racer_target_enc']
    cat_features = ['venue_code', 'racer_id']
    target = 'target'
    rank_target = 'rank_target'
    
    # Train/Val Split (Time-series split, shuffle=False)
    # Ensure races are not split between train/val by sorting correctly
    df = df.sort_values(['race_id', 'boat_number']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Calculate group sizes for LambdaRank
    train_groups = train_df.groupby('race_id', sort=False)['race_id'].count().values
    val_groups = val_df.groupby('race_id', sort=False)['race_id'].count().values
    
    X_train = train_df[features]
    y_train = train_df[target]
    y_train_rank = train_df[rank_target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    y_val_rank = val_df[rank_target]
    
    # 1. LightGBM (LambdaRank)
    model_lgb = train_lgb(X_train, y_train_rank, X_val, y_val_rank, train_groups, val_groups, cat_features)
    lgb_val_scores = model_lgb.predict(X_val)
    # Calibrate: Ranking scores to pseudo-probabilities via Softmax per race
    lgb_val_preds = softmax_calibration(lgb_val_scores, val_groups)
    print(f"LGB (LambdaRank calibrated) LogLoss: {log_loss(y_val, lgb_val_preds):.4f}")
    
    # 2. CatBoost (Binary classification)
    model_cb = train_catboost(X_train, y_train, X_val, y_val, cat_features)
    cb_val_preds = model_cb.predict_proba(X_val)[:, 1]
    print(f"CB LogLoss: {log_loss(y_val, cb_val_preds):.4f}")
    
    # 3. Alternative Model
    model_alt, X_val_alt, alt_scaler = train_alt_model(X_train, y_train, X_val, y_val)
    alt_val_preds = model_alt.predict_proba(X_val_alt)[:, 1]
    print(f"Alt-Model LogLoss: {log_loss(y_val, alt_val_preds):.4f}")

    # --- Stacking ---
    print("\n--- Meta-Stacking (V3.1: LGBM(Rank)+CB+AltModel) ---")
    X_meta = np.vstack([lgb_val_preds, cb_val_preds, alt_val_preds]).T
    
    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_val)
    
    final_preds = meta_model.predict_proba(X_meta)[:, 1]
    print(f"Stacked Model (V3.1) LogLoss: {log_loss(y_val, final_preds):.4f}")
    print(f"Stacked Model (V3.1) Accuracy: {accuracy_score(y_val, (final_preds > 0.5).astype(int)):.4f}")
    
    # Save Models
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    with open(os.path.join(MODEL_DIR, 'racer_mapping.pkl'), 'wb') as f:
        pickle.dump(racer_mapping, f)
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
